import copy
import json
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark, TaskInstance

from .livecodebench_utils import (
    lcb_run,
    lcb_run_test_cases,
    lcb_run_test_sets,
    map_to_example,
    post_process_code,
    translate_private_test_cases,
)

HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE")
OFFICIAL_DATASET_REPO = "livecodebench/code_generation_lite"
if not HF_HUB_CACHE:
    print(
        "WARNING: HF_HUB_CACHE environment variable is not set, using default cache directory ~/.cache/huggingface/hub for LiveCodeBench benchmark"
    )


def has_code(response):
    pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"
    # Use re.DOTALL to match multiline content inside backticks
    matches = re.findall(pattern, response, re.DOTALL)
    return matches


# Calculate mean and standard error for all metrics
def calc_stats(values):
    arr = np.asarray(values, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() == 0:  # all NaNs → undefined; return 0,0
        return 0.0, 0.0
    mean = arr[mask].mean()
    if mask.sum() < 2:
        return mean, 0.0
    stderr = np.std(arr[mask], ddof=1) / np.sqrt(mask.sum())
    return mean, stderr


class LiveCodeBenchBenchmark(BaseBenchmark):
    """
    LiveCodeBench Benchmark for evaluating the math reasoning of LLMs.

    Follows the evaluation logic of hendrycks_math answer extraction.
    """

    def __init__(
        self,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        version: Union[str, int] = "v2",
        dataset_repo: str = OFFICIAL_DATASET_REPO,
        dataset_split: str = "test",
        cache_dir: Optional[str] = HF_HUB_CACHE,
        contest_months: Optional[List[str]] = None,
        n_repeat: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize LiveCodeBench benchmark.

        Args:
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            version: LiveCodeBench release version, e.g. 2, "v2", "release_v2", 5, or 6.
            dataset_repo: Hugging Face dataset repo to load from.
            dataset_split: Split name to load.
            cache_dir: Dataset cache directory.
            contest_months: Optional YYYY-MM prefixes to filter by contest date.
            n_repeat: Optional override for how many times each sample is generated.
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.debug = debug
        self.max_new_tokens = max_tokens
        self.seed = seed
        self.dataset_repo = dataset_repo
        self.dataset_split = dataset_split
        self.cache_dir = cache_dir
        self.contest_months = set(contest_months) if contest_months else None
        self.version_tag = self._coerce_version_tag(version)
        self.n_repeat = n_repeat if n_repeat is not None else self._default_repeat_count(self.version_tag)

    @staticmethod
    def _coerce_version_tag(version: Union[str, int]) -> str:
        if isinstance(version, int):
            return f"v{version}"
        return str(version).strip()

    @staticmethod
    def _default_repeat_count(version_tag: str) -> int:
        return 6 if version_tag in {"2", "v2", "release_v2"} else 3

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and temporary directory,
            or None for non-primary ranks
        """
        examples_dataset = self.load_questions()
        # Convert the dataset object to a list
        examples = list(examples_dataset)
        if self.debug:
            examples = examples[:10]

        all_outputs = []

        for i in range(self.n_repeat):
            all_instances = []
            seed = [s + i for s in self.seed]

            for idx, example in enumerate(examples):
                # Type check for debugging purposes
                if not isinstance(example, dict):
                    self.logger.error(f"Example at index {idx} is not a dict. Type: {type(example)}, Value: {example}")
                    continue

                if example["is_stdin"]:
                    prompt_text = (
                        "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."
                        + example["prompt"]
                    )
                else:
                    prompt_text = (
                        "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."
                        + example["prompt"]
                    )
                messages = [{"role": "user", "content": prompt_text}]

                templated_messages = self._prepare_messages(messages, model)

                instance = Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            "do_sample": False,
                            "max_new_tokens": self.max_new_tokens,
                            "temperature": 0.7,
                            "seed": seed,
                        },
                    ),
                    idx,
                )
                instance.repeat_idx = i
                all_instances.append(instance)

            # Generate model responses
            self.logger.info("Generating responses for LiveCodeBench...")
            outputs = self.compute(model, all_instances)
            all_outputs.append(outputs)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        examples_list = []

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            example["model_answers"] = [has_code(o) for o in outputs]
            examples_list.append(example)

        return {"examples": examples_list}

    @staticmethod
    def check_correctness(problem: Dict, completion: str, timeout: float, is_extracted: bool = False) -> Dict:
        """
        Evaluates the functional correctness of a completion by running the test
        suite provided in the problem.

        :param completion_id: an optional completion ID so we can match
            the results later even if execution finishes asynchronously.
        """
        result_list = lcb_run(problem, completion, timeout, is_extracted)
        details = [r[0] for r in result_list]
        all_passed = all(details)

        result = ""
        if result_list and all_passed:
            result = "passed"

        return result == "passed"

    @staticmethod
    def _parse_public_test_cases(public_test_cases: Any) -> List[Dict[str, Any]]:
        if public_test_cases is None:
            return []
        if isinstance(public_test_cases, str):
            return json.loads(public_test_cases)
        return list(public_test_cases)

    def evaluate_public_test_cases(
        self, example: Dict[str, Any], completion: str, timeout: float, is_extracted: bool
    ) -> List[Dict[str, Any]]:
        public_test_cases = self._parse_public_test_cases(example.get("public_test_cases"))
        if not public_test_cases:
            return []

        results = lcb_run_test_cases(public_test_cases, completion, timeout, is_extracted)
        return [
            {
                "test_index": idx,
                "test_case": test_case,
                "passed": passed,
                "details": details,
                "output": output,
                "time_elapsed": time_elapsed,
            }
            for idx, (test_case, (passed, details, output, time_elapsed)) in enumerate(zip(public_test_cases, results))
        ]

    @staticmethod
    def _format_test_results(test_cases: List[Dict[str, Any]], results: List[tuple]) -> List[Dict[str, Any]]:
        return [
            {
                "test_index": idx,
                "test_case": test_case,
                "passed": passed,
                "details": details,
                "output": output,
                "time_elapsed": time_elapsed,
            }
            for idx, (test_case, (passed, details, output, time_elapsed)) in enumerate(zip(test_cases, results))
        ]

    @staticmethod
    def _extract_completion_code(raw_output: str) -> Optional[str]:
        code_blocks = has_code(raw_output)
        if code_blocks:
            return post_process_code(code_blocks[-1])
        stripped = raw_output.strip()
        return stripped or None

    @staticmethod
    def _single_sample_metrics(example: Dict[str, Any], response_entry: Dict[str, Any]) -> Dict[str, Any]:
        solved = int(response_entry["correctness"])
        accuracy = float(solved)
        difficulty = example.get("difficulty")
        return {
            "accuracy_avg": accuracy,
            "accuracy_std_err": 0.0,
            "num_total": 1,
            "solved_avg": accuracy,
            "num_repeat": 1,
            "raw_metrics": [
                {
                    "total_correct": solved,
                    "total_finish": 1,
                    "accuracy": accuracy,
                    "per_difficulty_correct": {difficulty: solved},
                    "per_difficulty_total": {difficulty: 1},
                }
            ],
            "run_stats": [{"repetition": 1, "num_total": 1, "num_solved": solved, "accuracy": accuracy}],
            "examples": [response_entry],
        }

    def evaluate_task_instance(self, task_instance: TaskInstance, raw_output: str) -> Dict[str, Any]:
        example = copy.deepcopy(task_instance.doc)
        code = self._extract_completion_code(raw_output)
        response_entry = {
            "content": code,
            "difficulty": example.get("difficulty"),
            "correctness": False,
            "reason": "Does not contain code component." if code is None else "Code is incorrect.",
            "public_test_results": [],
            "private_test_results": [],
        }

        if code is not None:
            public_test_cases = self._parse_public_test_cases(example.get("public_test_cases"))
            private_test_cases = example.get("test") or []
            is_extracted = not example.get("is_stdin", False)
            result_sets = lcb_run_test_sets(
                private_test_cases=private_test_cases,
                public_test_cases=public_test_cases,
                completion=code,
                timeout=6,
                is_extracted=is_extracted,
            )
            private_results = result_sets["private"]
            public_results = result_sets["public"]
            correctness = bool(private_results) and all(passed for passed, *_ in private_results)
            response_entry.update(
                {
                    "correctness": correctness,
                    "reason": "" if correctness else "Code is incorrect.",
                    "public_test_results": self._format_test_results(public_test_cases, public_results),
                    "private_test_results": self._format_test_results(private_test_cases, private_results),
                }
            )

        return {
            "supported": True,
            "raw_output": raw_output,
            "result": self._single_sample_metrics(example, response_entry),
            "public_test_results": response_entry["public_test_results"],
            "private_test_results": response_entry["private_test_results"],
        }

    def evaluate_single_example(self, example):
        """Helper function to evaluate a single example"""
        try:
            response_entry = {
                "content": example["model_answer"],
                "difficulty": example["difficulty"],
                "correctness": None,
                "reason": None,
                "public_test_results": [],
            }

            code_filter_result = example["model_answer"]

            if not code_filter_result or len(code_filter_result) == 0:
                response_entry["correctness"] = False
                response_entry["reason"] = "Does not contain code component."
                return response_entry

            try:
                last_code = code_filter_result[-1]
                problem_to_check = copy.deepcopy(example)

                # Add debugging
                self.logger.debug(f"Evaluating {example['difficulty']} problem...")

                # Add timeout handling
                curr_res = self.check_correctness(
                    problem=problem_to_check,
                    completion=post_process_code(last_code),
                    timeout=6,
                    is_extracted=not problem_to_check["is_stdin"],
                )
                response_entry["public_test_results"] = self.evaluate_public_test_cases(
                    example=problem_to_check,
                    completion=post_process_code(last_code),
                    timeout=6,
                    is_extracted=not problem_to_check["is_stdin"],
                )

                # Log the result
                self.logger.debug(f"Result for {example['difficulty']}: {curr_res}")

                response_entry["correctness"] = curr_res
                response_entry["reason"] = "" if curr_res else "Code is incorrect."

            except Exception as e:
                self.logger.error(f"Error evaluating {example['difficulty']} example: {str(e)}")
                response_entry["correctness"] = False
                response_entry["reason"] = f"Evaluation error: {str(e)}"

            return response_entry

        except Exception as outer_e:
            self.logger.error(f"Outer error in evaluate_single_example: {str(outer_e)}")
            return {
                "content": example.get("model_answer"),
                "difficulty": example.get("difficulty"),
                "correctness": False,
                "reason": f"Critical error: {str(outer_e)}",
                "public_test_results": [],
            }

    def evaluate_responses(self, responses: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions in parallel using threads."""
        # Handle None result from non-primary ranks
        if responses is None:
            return None

        self.logger.info(f"Evaluating {len(responses['examples'])} examples...")
        self.logger.warning(f"Expect some output leaks from the code / test execution into stdout")

        # First, organize completions by repeat index
        examples_by_repeat = defaultdict(list)
        for example in responses["examples"]:
            for i, (output, answers) in enumerate(zip(example["model_outputs"], example["model_answers"])):
                # Create a copy of the original example and update with the specific completion
                example_copy = example.copy()  # Make a shallow copy of the example
                example_copy["model_answer"] = answers
                example_copy["model_output"] = output
                # Remove the lists of all outputs/answers to avoid confusion
                example_copy.pop("model_outputs", None)
                example_copy.pop("model_answers", None)
                examples_by_repeat[i].append(example_copy)

        # Evaluate each set of completions separately
        all_metrics = []
        run_stats = []
        num_questions = len(responses["examples"])

        for repeat_idx, examples in examples_by_repeat.items():
            # Use ThreadPoolExecutor with limited concurrency
            results = []
            with ThreadPoolExecutor(max_workers=32) as executor:
                future_to_example = {}
                for i, example in enumerate(examples):
                    future = executor.submit(self.evaluate_single_example, example)
                    future_to_example[future] = (i, example)

                # Collect results as they complete
                results = [None] * len(examples)
                for future in as_completed(future_to_example):
                    idx, example = future_to_example[future]
                    try:
                        result = future.result()
                        results[idx] = (result, example)
                    except Exception as e:
                        self.logger.error(f"Future error for example {idx}: {str(e)}")
                        results[idx] = (
                            {
                                "content": example["model_answer"],
                                "difficulty": example["difficulty"],
                                "correctness": False,
                                "reason": f"Future error: {str(e)}",
                            },
                            example,
                        )

            # Calculate metrics for this repeat
            total_correct = sum(1 for result, _ in results if result["correctness"])
            total_finish = len(results)

            per_difficulty_correct = defaultdict(int)
            per_difficulty_total = defaultdict(int)

            for result, example in results:
                per_difficulty_correct[example["difficulty"]] += result["correctness"]
                per_difficulty_total[example["difficulty"]] += 1

            metrics = {
                "total_correct": total_correct,
                "total_finish": total_finish,
                "accuracy": total_correct / total_finish,
                "per_difficulty_correct": dict(per_difficulty_correct),
                "per_difficulty_total": dict(per_difficulty_total),
            }

            # Add per-difficulty accuracies
            for difficulty in per_difficulty_correct.keys():
                metrics[f"accuracy_{difficulty}"] = (
                    per_difficulty_correct[difficulty] / per_difficulty_total[difficulty]
                )

            all_metrics.append(metrics)

            # Add to run_stats for precomputed_hf_lm.py compatibility
            run_stats.append(
                {
                    "repetition": repeat_idx + 1,
                    "num_total": total_finish,
                    "num_solved": total_correct,
                    "accuracy": total_correct / total_finish,
                }
            )

        final_metrics = {}

        # Calculate stats for overall accuracy
        acc_values = [m["accuracy"] for m in all_metrics]
        mean_acc, stderr_acc = calc_stats(acc_values)
        final_metrics["accuracy_avg"] = mean_acc
        final_metrics["accuracy_std_err"] = stderr_acc
        self.logger.info(f"Overall accuracy: {mean_acc:.2%} ± {stderr_acc:.2%}")

        # Calculate stats for each difficulty level
        difficulties = all_metrics[0]["per_difficulty_correct"].keys()
        for diff in difficulties:
            acc_values = [m[f"accuracy_{diff}"] for m in all_metrics]
            mean_acc, stderr_acc = calc_stats(acc_values)
            final_metrics[f"accuracy_{diff}_avg"] = mean_acc
            final_metrics[f"accuracy_{diff}_std_err"] = stderr_acc

        # Log results
        for diff in difficulties:
            mean = final_metrics[f"accuracy_{diff}_avg"]
            stderr = final_metrics[f"accuracy_{diff}_std_err"]
            self.logger.info(f"Accuracy {diff}: {mean:.2%} ± {stderr:.2%}")

        # Include raw results and examples in final metrics
        final_metrics["raw_metrics"] = all_metrics
        final_metrics["examples"] = [result for result, _ in results]  # Include last run's examples

        # Add compatibility with precomputed_hf_lm.py
        solved_avg = np.mean([result["num_solved"] for result in run_stats])
        final_metrics.update(
            {
                "num_total": num_questions,
                "solved_avg": solved_avg,
                "run_stats": run_stats,
                "num_repeat": self.n_repeat,
            }
        )

        return final_metrics

    def load_questions(self) -> Dataset:
        """Load LiveCodeBench questions from source."""
        self.logger.info(f"Loading LiveCodeBench questions from {self.dataset_repo} for {self.version_tag}...")
        cpu_count = os.cpu_count()
        dataset_kwargs = {
            "split": self.dataset_split,
            "cache_dir": self.cache_dir,
            "trust_remote_code": True,
        }
        if self.dataset_repo == OFFICIAL_DATASET_REPO:
            dataset_kwargs["version_tag"] = self.version_tag

        ds = load_dataset(self.dataset_repo, **dataset_kwargs)
        if self.contest_months is not None:
            ds = ds.filter(lambda example: example["contest_date"][:7] in self.contest_months)

        # Avoids "pyarrow.lib.ArrowInvalid: offset overflow while concatenating arrays" when mapping
        processed_shards = []
        num_shards = 4
        for i in range(num_shards):
            shard = ds.shard(num_shards=num_shards, index=i)
            shard = shard.map(
                lambda example: {"private_test_cases": translate_private_test_cases(example["private_test_cases"])},
                num_proc=cpu_count,
            )
            shard = shard.map(map_to_example, remove_columns=ds.column_names)
            processed_shards.append(shard)
        ds = concatenate_datasets(processed_shards)
        return ds
