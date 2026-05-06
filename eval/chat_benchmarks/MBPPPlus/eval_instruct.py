from typing import Dict, List, Any, Optional, Generator
import ast
import json
import multiprocessing
import os
import re
import tempfile
import time
from pathlib import Path
from tqdm import tqdm
import logging

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from .mbpp_plus.evaluation import (
    IMPORT_HELPER,
    evaluate_functional_correctness,
    get_task_timeout,
    normalize_mbpp_test,
    read_dataset,
)
from .mbpp_plus.execution import TimeoutException, reliability_guard, swallow_io, time_limit
from .utils.utils import extract_generation_code, language_settings
from eval.task import BaseBenchmark
from eval.task import TaskInstance


def _safe_repr(value: Any, max_length: int = 500) -> str:
    text = repr(value)
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def _get_mp_context():
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return multiprocessing.get_context()


def _run_public_feedback_worker(generation: str, public_tests: List[str], timeout: float, result_list):
    reliability_guard()
    namespace = {}
    try:
        with swallow_io():
            with time_limit(timeout):
                exec("\n".join(IMPORT_HELPER["python"]) + "\n" + generation, namespace)
    except TimeoutException:
        for idx, test_case in enumerate(public_tests):
            result_list.append(
                {
                    "test_index": idx,
                    "test_case": test_case,
                    "passed": False,
                    "details": "Timed out while preparing generated code.",
                    "output": None,
                    "time_elapsed": float("inf"),
                }
            )
        return
    except BaseException as exc:
        for idx, test_case in enumerate(public_tests):
            result_list.append(
                {
                    "test_index": idx,
                    "test_case": test_case,
                    "passed": False,
                    "details": f"Generated code failed before running tests: {exc}",
                    "output": None,
                    "time_elapsed": 0.0,
                }
            )
        return

    for idx, test_case in enumerate(public_tests):
        start = time.time()
        try:
            with swallow_io():
                with time_limit(timeout):
                    exec(test_case, namespace)
            result_list.append(
                {
                    "test_index": idx,
                    "test_case": test_case,
                    "passed": True,
                    "details": "Public test passed.",
                    "output": None,
                    "time_elapsed": time.time() - start,
                }
            )
        except TimeoutException:
            result_list.append(
                {
                    "test_index": idx,
                    "test_case": test_case,
                    "passed": False,
                    "details": "Public test timed out.",
                    "output": None,
                    "time_elapsed": float("inf"),
                }
            )
        except AssertionError as exc:
            result_list.append(
                {
                    "test_index": idx,
                    "test_case": test_case,
                    "passed": False,
                    "details": f"Public test failed: AssertionError{': ' + str(exc) if str(exc) else ''}",
                    "output": None,
                    "time_elapsed": time.time() - start,
                }
            )
        except BaseException as exc:
            result_list.append(
                {
                    "test_index": idx,
                    "test_case": test_case,
                    "passed": False,
                    "details": f"Public test failed with error: {exc}",
                    "output": None,
                    "time_elapsed": time.time() - start,
                }
            )


def _run_private_feedback_worker(
    generation: str,
    setup_code: str,
    actual_expr: str,
    expected_expr: str,
    timeout: float,
    result_list,
):
    reliability_guard()
    namespace = {}
    try:
        with swallow_io():
            with time_limit(timeout):
                exec(generation + "\n" + setup_code, namespace)
    except TimeoutException:
        result_list.append(
            {
                "test_index": 0,
                "test_case": None,
                "passed": False,
                "details": "Timed out while preparing generated code and private tests.",
                "output": None,
                "expected": None,
                "time_elapsed": float("inf"),
            }
        )
        return
    except BaseException as exc:
        result_list.append(
            {
                "test_index": 0,
                "test_case": None,
                "passed": False,
                "details": f"Generated code or private test setup failed: {exc}",
                "output": None,
                "expected": None,
                "time_elapsed": 0.0,
            }
        )
        return

    inputs = namespace.get("inputs", [])
    results = namespace.get("results")
    assertion = namespace.get("assertion")
    for idx, inp in enumerate(inputs):
        start = time.time()
        local_namespace = dict(namespace)
        local_namespace["i"] = idx
        local_namespace["inp"] = inp
        if results is not None:
            local_namespace["exp"] = results[idx]

        output = None
        expected = None
        try:
            with swallow_io():
                with time_limit(timeout):
                    expected = eval(expected_expr, local_namespace)
                    output = eval(actual_expr, local_namespace)
                    assertion(output, expected, 0)
            result_list.append(
                {
                    "test_index": idx,
                    "test_case": {"input": inp},
                    "passed": True,
                    "details": f"Private test passed with output {_safe_repr(output)}.",
                    "output": output,
                    "expected": expected,
                    "time_elapsed": time.time() - start,
                }
            )
        except TimeoutException:
            result_list.append(
                {
                    "test_index": idx,
                    "test_case": {"input": inp},
                    "passed": False,
                    "details": "Private test timed out.",
                    "output": output,
                    "expected": expected,
                    "time_elapsed": float("inf"),
                }
            )
        except AssertionError as exc:
            result_list.append(
                {
                    "test_index": idx,
                    "test_case": {"input": inp},
                    "passed": False,
                    "details": str(exc) or f"Expected {_safe_repr(expected)}, but got {_safe_repr(output)}.",
                    "output": output,
                    "expected": expected,
                    "time_elapsed": time.time() - start,
                }
            )
        except BaseException as exc:
            result_list.append(
                {
                    "test_index": idx,
                    "test_case": {"input": inp},
                    "passed": False,
                    "details": f"Private test failed with error: {exc}",
                    "output": output,
                    "expected": expected,
                    "time_elapsed": time.time() - start,
                }
            )


class MBPPPlusBenchmark(BaseBenchmark):
    """
    MBPPPlus benchmark for evaluating code generation capabilities across different languages.
    """

    DEFAULT_TASK_TIMEOUTS = {
        255: 120.0,
        271: 30.0,
        392: 30.0,
        599: 30.0,
        630: 30.0,
    }

    def __init__(
        self,
        data_dir: str = "eval/chat_benchmarks/MBPPPlus/data",
        num_workers: int = 8,
        timeout: float = 3.0,
        task_timeouts: Optional[Dict[int, float]] = None,
        debug: bool = False,
        max_tokens: int = 1024,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize MBPPPlus benchmark.

        Args:
            data_dir: Directory containing MBPPPlus datasets
            max_tokens: Maximum number of tokens for generation
            num_workers: Number of workers for parallel evaluation
            timeout: Timeout for code execution
            task_timeouts: Optional per-task timeout overrides for slow private tests
            debug: If True, only evaluate first 2 examples
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_dir = self.resolve_asset_path(data_dir)
        self.max_tokens = max_tokens
        self.num_workers = num_workers
        self.timeout = timeout
        self.task_timeouts = dict(self.DEFAULT_TASK_TIMEOUTS if task_timeouts is None else task_timeouts)
        self.debug = debug
        self.num_examples = 3
        self.start_idx = 0
        self.end_idx = 500

    def format_test_example(self, question: str, tests: List[str], code: Optional[str] = None) -> str:
        """Format a single test example."""
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(question.strip(), "\n".join(tests))
        if code:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
        return prompt

    def read_test_examples(self, data_path: str) -> Generator[Dict[str, str], None, None]:
        """
        Read and format test examples from data file.

        Args:
            data_path: Path to the data file

        Yields:
            Dictionary containing task_id and formatted prompt
        """
        try:
            with open(data_path, "r") as f:
                examples = [json.loads(x) for x in f]
            self.logger.info(f"Loaded {len(examples)} examples from {data_path}")

            examples_str = []
            for i in range(1, self.num_examples + 1):
                ex = examples[i]
                example_prompt = "- Example {}:\n{}".format(
                    i, self.format_test_example(ex["prompt"], ex["test_list"], ex["code"])
                )
                examples_str.append(example_prompt)

            eval_range = range(self.start_idx, min(self.end_idx, len(examples)))
            if self.debug:
                eval_range = list(eval_range)[:2]
                self.logger.info(f"Debug mode: using 2 examples")

            for i in eval_range:
                ex = examples[i]
                prompt = self.format_test_example(ex["prompt"], ex["test_list"])

                prompt_with_shots = """
Please refer the given examples and generate a python function for my problem.
Examples are listed as follows:
{}

Here is my problem:
{}
""".strip().format(
                    "\n\n".join(examples_str), prompt
                )

                yield {"task_id": ex["task_id"], "prompt": prompt_with_shots}

        except Exception as e:
            self.logger.error(f"Error reading examples: {str(e)}")
            raise

    def extract_code(self, completion: str) -> str:
        """Extract code block from model completion."""
        try:
            code_block = re.findall(r"```python\n(.*?)```", completion, re.DOTALL | re.IGNORECASE)[0]
            return code_block
        except Exception as e:
            self.logger.warning(f"Failed to extract code block, using full completion.\nError: {str(e)}")
            return completion

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate code completions using the provided model.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing generated responses and temporary directory,
            or None for non-primary ranks
        """
        try:
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = temp_dir_obj.name

            problem_file = os.path.join(self.data_dir, "mbppplus.jsonl")
            examples = list(self.read_test_examples(problem_file))
            self.logger.info(f"Processing {len(examples)} examples")

            all_instances = []
            for idx, example in enumerate(examples):
                try:
                    inputs = self._prepare_messages([{"role": "user", "content": example["prompt"]}], model)

                    all_instances.append(
                        Instance(
                            "generate_until",
                            example,
                            (
                                inputs,
                                {
                                    "max_new_tokens": self.max_tokens,
                                    "do_sample": False,
                                },
                            ),
                            idx,
                        )
                    )
                except Exception as e:
                    self.logger.error(f"Error preparing instance {idx}: {str(e)}")
                    continue

            self.logger.info("Generating responses for MBPPPlus...")
            outputs = self.compute(model, all_instances)

            # Return None early for non-primary ranks
            if model.rank != 0:
                return None

            generated_examples = []
            for example, output in zip(examples, outputs):
                try:
                    example_with_output = example.copy()
                    example_with_output["gpt_completion"] = output
                    example_with_output["generation"] = self.extract_code(output)
                    generated_examples.append(example_with_output)
                except Exception as e:
                    self.logger.error(f"Error processing output for {example['task_id']}: {str(e)}")
                    continue

            output_path = os.path.join(temp_dir, "generated_python.jsonl")
            with open(output_path, "w", encoding="utf-8") as fw:
                for ex in generated_examples:
                    fw.write(json.dumps(ex) + "\n")

            self.logger.info(f"Saved {len(generated_examples)} examples to {output_path}")

            return {
                "temp_dir_obj": temp_dir_obj,
                "num_examples": len(generated_examples),
                "total_examples": len(examples),
            }

        except Exception as e:
            self.logger.error(f"Error in generate_responses: {str(e)}")
            raise

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the generated code completions.

        Args:
            results: Dictionary containing generation results

        Returns:
            Dictionary containing evaluation metrics
        """
        # Handle None result from non-primary ranks
        if results is None:
            return None

        temp_dir_obj = results["temp_dir_obj"]
        temp_dir = temp_dir_obj.name

        evaluation_results = {}

        problem_file = os.path.join(self.data_dir, f"mbppplus.jsonl")
        temp_file_path = os.path.join(temp_dir, f"generated_python.jsonl")

        if not os.path.exists(temp_file_path):
            self.logger.warning(f"Generated file not found: {temp_file_path}")

        result = evaluate_functional_correctness(
            input_file=temp_file_path,
            tmp_dir=temp_dir,
            n_workers=self.num_workers,
            timeout=self.timeout,
            problem_file=problem_file,
            language="python",
            is_mbpp=True,
            task_timeouts=self.task_timeouts,
        )

        for metric, value in result.items():
            evaluation_results[f"{metric}"] = value

        self.logger.info(f"Completed evaluation")

        temp_dir_obj.cleanup()
        return evaluation_results

    @staticmethod
    def _run_feedback_worker(target, args: tuple, timeout: float) -> List[Dict[str, Any]]:
        mp_context = _get_mp_context()
        manager = mp_context.Manager()
        result_list = manager.list()
        process = mp_context.Process(target=target, args=(*args, result_list))
        process.start()
        process.join(timeout + 1)
        if process.is_alive():
            process.kill()
            process.join()
            result_list.append(
                {
                    "test_index": len(result_list),
                    "test_case": None,
                    "passed": False,
                    "details": "Timed out while collecting test feedback.",
                    "output": None,
                    "time_elapsed": float("inf"),
                }
            )
        results = list(result_list)
        manager.shutdown()
        return results

    def evaluate_public_test_cases(
        self, example: Dict[str, Any], generation: str, timeout: float
    ) -> List[Dict[str, Any]]:
        public_tests = example.get("test_list") or []
        if not public_tests:
            return []
        return self._run_feedback_worker(
            _run_public_feedback_worker,
            (generation, public_tests, timeout),
            timeout * max(len(public_tests), 1),
        )

    @staticmethod
    def _split_private_feedback_test(test: str):
        test = normalize_mbpp_test(test)
        match = re.search(r"\nfor i,\s*(?:\(inp,\s*exp\)|inp)\s+in\s+enumerate\(.*?\):\n(?P<body>.*)$", test, re.DOTALL)
        if not match:
            return None, None, None

        setup_code = test[: match.start()]
        assertion_line = None
        for line in match.group("body").splitlines():
            stripped = line.strip()
            if stripped.startswith("assertion("):
                assertion_line = stripped
                break
        if assertion_line is None:
            return None, None, None

        call = ast.parse(assertion_line).body[0].value
        if not isinstance(call, ast.Call) or len(call.args) < 2:
            return None, None, None

        return setup_code, ast.unparse(call.args[0]), ast.unparse(call.args[1])

    def evaluate_private_test_cases(
        self, example: Dict[str, Any], generation: str, timeout: float
    ) -> List[Dict[str, Any]]:
        test = example.get("test")
        if not isinstance(test, str):
            return []

        setup_code, actual_expr, expected_expr = self._split_private_feedback_test(test)
        if setup_code is None:
            return [
                {
                    "test_index": 0,
                    "test_case": None,
                    "passed": False,
                    "details": "Private test feedback is unavailable for this test format.",
                    "output": None,
                    "expected": None,
                    "time_elapsed": 0.0,
                }
            ]

        return self._run_feedback_worker(
            _run_private_feedback_worker,
            (generation, setup_code, actual_expr, expected_expr, timeout),
            timeout * max(len(example.get("test_list") or []), 1) + 5,
        )

    def evaluate_task_instance(self, task_instance: TaskInstance, raw_output: str) -> Dict[str, Any]:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, "generated_python.jsonl")
                generation = self.extract_code(raw_output)
                sample = {
                    "task_id": task_instance.doc["task_id"],
                    "generation": generation,
                }
                with open(temp_file_path, "w", encoding="utf-8") as fw:
                    fw.write(json.dumps(sample) + "\n")

                task_timeout = get_task_timeout(task_instance.doc["task_id"], self.timeout, self.task_timeouts)
                result = evaluate_functional_correctness(
                    input_file=temp_file_path,
                    tmp_dir=temp_dir,
                    n_workers=self.num_workers,
                    timeout=self.timeout,
                    problem_file=os.path.join(self.data_dir, "mbppplus.jsonl"),
                    language="python",
                    is_mbpp=True,
                    k=[1],
                    task_timeouts=self.task_timeouts,
                )
                problem = read_dataset(os.path.join(self.data_dir, "mbppplus.jsonl"))[task_instance.doc["task_id"]]
                public_test_results = self.evaluate_public_test_cases(problem, generation, task_timeout)
                private_test_results = self.evaluate_private_test_cases(problem, generation, task_timeout)

            return {
                "supported": True,
                "raw_output": raw_output,
                "result": result,
                "public_test_results": public_test_results,
                "private_test_results": private_test_results,
            }
        except Exception as exc:
            return {
                "supported": False,
                "raw_output": raw_output,
                "reason": f"Benchmark evaluator could not score one sample: {exc}",
            }

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """
        Run the complete benchmark evaluation pipeline.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation results, or None for non-primary ranks
        """
        self.logger.info(f"Running MBPPPlus benchmark")
        try:
            generation_results = self.generate_responses(model)

            # If not primary rank, return None early
            if generation_results is None:
                return None

            evaluation_results = self.evaluate_responses(generation_results)
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
