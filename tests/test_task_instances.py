import contextlib
import io
import os
import re
import sys
import tempfile
import types
import unittest
from unittest import mock


def install_lm_eval_stubs():
    try:
        import lm_eval  # noqa: F401

        return
    except ImportError:
        pass

    class Instance:
        def __init__(self, request_type, doc, args, idx):
            self.request_type = request_type
            self.doc = doc
            self.args = args
            self.idx = idx

    class LM:
        pass

    class OpenAIChatCompletion:
        pass

    class OpenAICompletionsAPI:
        pass

    class VLLM:
        pass

    lm_eval = types.ModuleType("lm_eval")
    api = types.ModuleType("lm_eval.api")
    instance_mod = types.ModuleType("lm_eval.api.instance")
    model_mod = types.ModuleType("lm_eval.api.model")
    models_mod = types.ModuleType("lm_eval.models")
    tasks_mod = types.ModuleType("lm_eval.tasks")
    hendrycks_math_mod = types.ModuleType("lm_eval.tasks.hendrycks_math")
    hendrycks_math_utils_mod = types.ModuleType("lm_eval.tasks.hendrycks_math.utils")
    openai_mod = types.SimpleNamespace(
        OpenAIChatCompletion=OpenAIChatCompletion,
        OpenAICompletionsAPI=OpenAICompletionsAPI,
    )
    vllm_mod = types.SimpleNamespace(VLLM=VLLM)

    instance_mod.Instance = Instance
    model_mod.LM = LM
    models_mod.openai_completions = openai_mod
    models_mod.vllm_causallms = vllm_mod
    hendrycks_math_utils_mod.is_equiv = lambda left, right: str(left) == str(right)
    hendrycks_math_utils_mod.last_boxed_only_string = lambda text: text
    hendrycks_math_utils_mod.remove_boxed = lambda text: text.replace("\\boxed{", "").replace("}", "")

    lm_eval.api = api
    lm_eval.models = models_mod
    lm_eval.tasks = tasks_mod
    api.instance = instance_mod
    api.model = model_mod
    tasks_mod.hendrycks_math = hendrycks_math_mod
    hendrycks_math_mod.utils = hendrycks_math_utils_mod

    sys.modules["lm_eval"] = lm_eval
    sys.modules["lm_eval.api"] = api
    sys.modules["lm_eval.api.instance"] = instance_mod
    sys.modules["lm_eval.api.model"] = model_mod
    sys.modules["lm_eval.models"] = models_mod
    sys.modules["lm_eval.tasks"] = tasks_mod
    sys.modules["lm_eval.tasks.hendrycks_math"] = hendrycks_math_mod
    sys.modules["lm_eval.tasks.hendrycks_math.utils"] = hendrycks_math_utils_mod


def install_torch_stubs():
    try:
        import torch  # noqa: F401
        import torch.distributed  # noqa: F401

        return
    except ImportError:
        pass

    torch = types.ModuleType("torch")
    distributed = types.ModuleType("torch.distributed")

    torch.manual_seed = lambda seed: None
    distributed.all_gather_object = lambda all_results, results: None

    sys.modules["torch"] = torch
    sys.modules["torch.distributed"] = distributed


def install_numpy_stubs():
    try:
        import numpy  # noqa: F401

        return
    except ImportError:
        pass

    numpy = types.ModuleType("numpy")
    numpy.random = types.SimpleNamespace(seed=lambda seed: None)
    sys.modules["numpy"] = numpy


install_lm_eval_stubs()
install_torch_stubs()
install_numpy_stubs()

from lm_eval.api.instance import Instance

from eval.task import BaseBenchmark, resolve_package_asset_path


class SyntheticBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.examples = [
            {"id": "sample-1", "answer": "yes"},
            {"id": "sample-2", "answer": "no"},
        ]

    def generate_responses(self, model):
        instances = []
        for idx, example in enumerate(self.examples):
            prompt = self._prepare_messages([{"role": "user", "content": f"Answer {example['id']}"}], model)
            instances.append(
                Instance(
                    "generate_until",
                    example,
                    (
                        prompt,
                        {
                            "max_new_tokens": 8,
                            "temperature": 0.0,
                            "seed": [0, 1, 2, 3],
                        },
                    ),
                    idx,
                )
            )
        outputs = self.compute(model, instances)
        if model.rank != 0:
            return None
        for example, output in zip(self.examples, outputs):
            example["model_output"] = output
        return {"examples": self.examples}

    def evaluate_responses(self, results):
        examples = results["examples"]
        solved = sum(example.get("model_output") == example["answer"] for example in examples)
        return {"num_total": len(examples), "num_solved": solved, "accuracy": solved / len(examples)}


class SyntheticGSM8KBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.examples = [
            {"id": "gsm-1", "question": "What is 40 + 2?", "answer": "42"},
            {"id": "gsm-2", "question": "What is 1,000 + 250?", "answer": "1250"},
        ]

    def generate_responses(self, model):
        instances = []
        for idx, example in enumerate(self.examples):
            prompt = self._prepare_messages(
                [
                    {
                        "role": "user",
                        "content": f"{example['question']}\nGive the final answer as #### <answer>.",
                    }
                ],
                model,
            )
            instances.append(
                Instance(
                    "generate_until",
                    example,
                    (prompt, {"max_new_tokens": 64, "temperature": 0.0}),
                    idx,
                )
            )
        outputs = self.compute(model, instances)
        if model.rank != 0:
            return None
        for example, output in zip(self.examples, outputs):
            example["model_output"] = output
            example["model_answer"] = self.extract_answer(output)
        return {"examples": self.examples}

    def evaluate_responses(self, results):
        examples = results["examples"]
        solved = sum(example.get("model_answer") == example["answer"] for example in examples)
        no_answer = sum(example.get("model_answer") == "" for example in examples)
        return {
            "num_total": len(examples),
            "num_solved": solved,
            "num_no_answer": no_answer,
            "accuracy": solved / len(examples),
        }

    def extract_answer(self, output):
        marker_matches = re.findall(r"####\s*([-+]?\$?[\d,]+(?:\.\d+)?)", output)
        if marker_matches:
            return self._normalize_number(marker_matches[-1])

        number_matches = re.findall(r"[-+]?\$?[\d,]+(?:\.\d+)?", output)
        if number_matches:
            return self._normalize_number(number_matches[-1])
        return ""

    @staticmethod
    def _normalize_number(value):
        return value.replace("$", "").replace(",", "").strip()


def has_code(response):
    return re.findall(r"```(?:python)?\n(.*?)```", response, re.DOTALL)


class SyntheticCodeBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.examples = [{"id": "code-1", "answer": "return 1"}]

    def generate_responses(self, model):
        example = self.examples[0]
        prompt = self._prepare_messages([{"role": "user", "content": "Write code."}], model)
        outputs = self.compute(
            model,
            [
                Instance(
                    "generate_until",
                    example,
                    (prompt, {"max_new_tokens": 32, "temperature": 0.0}),
                    0,
                )
            ],
        )
        if model.rank != 0:
            return None
        return {"examples": [{**example, "model_outputs": outputs, "model_answers": [has_code(outputs[0])]}]}

    def evaluate_responses(self, results):
        example = results["examples"][0]
        solved = example["model_answers"][0] == [example["answer"]]
        return {"num_total": 1, "num_solved": int(solved), "accuracy": float(solved)}


class TemplateModel:
    model = "template-model"
    model_args = {"model": "template-model"}

    def apply_chat_template(self, messages):
        return " ".join(f"{message['role']}: {message['content']}" for message in messages)


class AnsweringLM:
    rank = 0
    world_size = 1

    def apply_chat_template(self, messages):
        return messages

    def generate_until(self, requests):
        return [request.doc["answer"] for request in requests]


class FakeDataset:
    column_names = ["question_id"]

    def shard(self, num_shards, index):
        return self

    def map(self, fn, num_proc=None, remove_columns=None):
        return self


class TaskInstanceTests(unittest.TestCase):
    def test_task_instances_capture_generation_requests(self):
        benchmark = SyntheticBenchmark()

        tasks = benchmark.task_instances()

        self.assertEqual([task.id for task in tasks], ["sample-1", "sample-2"])
        self.assertEqual(tasks[0].request_type, "generate_until")
        self.assertEqual(tasks[0].prompt, [{"role": "user", "content": "Answer sample-1"}])
        self.assertEqual(tasks[0].generation_kwargs, {"max_new_tokens": 8, "temperature": 0.0})
        self.assertEqual(tasks[0].metadata["task_name"], "Synthetic")

    def test_task_instances_can_use_target_model_chat_template(self):
        benchmark = SyntheticBenchmark()

        task = benchmark.task_instances(model=TemplateModel())[0]

        self.assertEqual(task.prompt, "user: Answer sample-1")

    def test_task_instance_evaluates_raw_output_with_existing_evaluator(self):
        benchmark = SyntheticBenchmark()
        task = benchmark.task_instances()[0]

        result = task.evaluate("yes")

        self.assertIs(result["supported"], True)
        self.assertEqual(result["result"]["num_total"], 1)
        self.assertEqual(result["result"]["num_solved"], 1)
        self.assertEqual(result["result"]["accuracy"], 1.0)

    def test_synthetic_gsm8k_extraction_and_raw_output_evaluation(self):
        benchmark = SyntheticGSM8KBenchmark()
        first, second = benchmark.task_instances()

        self.assertEqual(first.prompt[0]["content"], "What is 40 + 2?\nGive the final answer as #### <answer>.")
        self.assertEqual(first.evaluate("Reasoning here. #### 42")["result"]["accuracy"], 1.0)
        self.assertEqual(first.evaluate("The final answer is 42.")["result"]["accuracy"], 1.0)
        self.assertEqual(second.evaluate("Compute it carefully: #### 1,250")["result"]["accuracy"], 1.0)
        self.assertEqual(first.evaluate("#### 41")["result"]["accuracy"], 0.0)
        self.assertEqual(first.evaluate("I cannot tell.")["result"]["num_no_answer"], 1)

    def test_task_instance_uses_module_level_has_code_when_available(self):
        benchmark = SyntheticCodeBenchmark()
        task = benchmark.task_instances()[0]

        result = task.evaluate("```python\nreturn 1```")

        self.assertIs(result["supported"], True)
        self.assertEqual(result["result"]["accuracy"], 1.0)

    def test_existing_run_benchmark_interface_still_works(self):
        benchmark = SyntheticBenchmark()

        with contextlib.redirect_stdout(io.StringIO()):
            result = benchmark.run_benchmark(AnsweringLM())

        self.assertEqual(result["num_total"], 2)
        self.assertEqual(result["num_solved"], 2)
        self.assertEqual(result["accuracy"], 1.0)

    def test_resolve_package_asset_path_maps_repo_style_path_to_installed_location(self):
        resolved = resolve_package_asset_path("eval/chat_benchmarks/AIME24/data/aime24.json")

        self.assertTrue(os.path.isabs(resolved))
        self.assertTrue(resolved.endswith("eval/chat_benchmarks/AIME24/data/aime24.json"))
        self.assertTrue(os.path.exists(resolved))

    def test_aime24_default_data_file_loads_outside_repo_cwd(self):
        from eval.chat_benchmarks.AIME24.eval_instruct import AIME24Benchmark

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                benchmark = AIME24Benchmark(debug=True)
                questions = benchmark.load_questions()
            finally:
                os.chdir(original_cwd)

        self.assertEqual(len(questions), 2)
        self.assertTrue(os.path.isabs(benchmark.data_file))
        self.assertTrue(benchmark.data_file.endswith("eval/chat_benchmarks/AIME24/data/aime24.json"))

    def test_math500_task_manager_and_task_instances_work(self):
        from eval.task import TaskManager

        task_manager = TaskManager(task_list=["MATH500"], debug=True)
        benchmark = task_manager.get_benchmark("MATH500")

        self.assertIsNotNone(benchmark)
        task = benchmark.task_instances()[0]

        self.assertEqual(task.request_type, "generate_until")
        self.assertEqual(task.generation_kwargs["max_new_tokens"], 32768)
        self.assertEqual(task.evaluate(r"\boxed{42}")["supported"], True)

    def test_mbppplus_task_manager_and_task_instances_work(self):
        from eval.task import TaskManager

        task_manager = TaskManager(task_list=["MBPPPlus"], debug=True)
        benchmark = task_manager.get_benchmark("MBPPPlus")

        self.assertIsNotNone(benchmark)
        task = benchmark.task_instances()[0]

        self.assertEqual(task.request_type, "generate_until")
        self.assertEqual(task.generation_kwargs["max_new_tokens"], 1024)
        result = task.evaluate("```python\ndef foo():\n    return 42\n```")
        self.assertTrue(result["supported"])
        self.assertIn("pass@1", result["result"])

    def test_mbppplus_process_humaneval_test_handles_string_test_field(self):
        from eval.chat_benchmarks.MBPPPlus.mbpp_plus.evaluation import process_humaneval_test

        sample = {"task_id": "task-1", "generation": "def f():\n    return 1"}
        problems = {"task-1": {"test": "assert f() == 1"}}

        test_code = process_humaneval_test(sample, problems, is_mbpp=True)

        self.assertEqual(test_code, "def f():\n    return 1\nassert f() == 1")

    def test_mbppplus_private_test_assertion_accepts_nested_numeric_tolerance(self):
        from eval.chat_benchmarks.MBPPPlus.mbpp_plus.evaluation import process_humaneval_test

        sample = {
            "task_id": "task-1",
            "generation": "def f():\n    return ((1.000000000000001,), 2 + 0j, 10**200, float('inf'))",
        }
        problems = {
            "task-1": {
                "test": """import numpy as np
from math import inf

def is_floats(x) -> bool:
    # check if it is float; List[float]; Tuple[float]
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)):
        return all(isinstance(i, float) for i in x)
    if isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False


def assertion(out, exp, atol):
    if atol == 0 and is_floats(exp):
        atol = 1e-6
    if out != exp and atol != 0:
        assert np.allclose(out, exp, rtol=1e-07, atol=atol)
    else:
        assert out == exp, f"out: {out}, exp: {exp}"

assertion(f(), ((1.0,), 2 + 1e-15j, 10**200, float('inf')), 0)
"""
            }
        }

        test_code = process_humaneval_test(sample, problems, is_mbpp=True)

        self.assertIn("def numeric_close", test_code)
        exec(test_code, {})

    def test_mbppplus_task_timeout_overrides_are_task_specific(self):
        from eval.chat_benchmarks.MBPPPlus.eval_instruct import MBPPPlusBenchmark
        from eval.chat_benchmarks.MBPPPlus.mbpp_plus.evaluation import get_task_timeout

        benchmark = MBPPPlusBenchmark(debug=True)

        self.assertEqual(get_task_timeout(255, benchmark.timeout, benchmark.task_timeouts), 120.0)
        self.assertEqual(get_task_timeout("255", benchmark.timeout, benchmark.task_timeouts), 120.0)
        self.assertEqual(get_task_timeout(630, benchmark.timeout, benchmark.task_timeouts), 30.0)
        self.assertEqual(get_task_timeout(1, benchmark.timeout, benchmark.task_timeouts), 3.0)

    def test_mbppplus_reliability_guard_keeps_os_putenv_for_numpy_imports(self):
        from eval.chat_benchmarks.MBPPPlus.mbpp_plus import execution as mbpp_exec

        original_putenv = os.putenv
        original_environ = os.environ.copy()
        original_os_attrs = {
            name: getattr(os, name)
            for name in [
                "kill",
                "system",
                "remove",
                "removedirs",
                "rmdir",
                "fchdir",
                "setuid",
                "fork",
                "forkpty",
                "killpg",
                "rename",
                "renames",
                "truncate",
                "replace",
                "unlink",
                "fchmod",
                "fchown",
                "chmod",
                "chown",
                "chroot",
                "lchflags",
                "lchmod",
                "lchown",
                "getcwd",
                "chdir",
            ]
            if hasattr(os, name)
        }

        try:
            mbpp_exec.reliability_guard()
            self.assertIs(os.putenv, original_putenv)
        finally:
            os.putenv = original_putenv
            os.environ.clear()
            os.environ.update(original_environ)
            for name, value in original_os_attrs.items():
                setattr(os, name, value)

    def test_livecodebench_version_passthrough_and_repeat_defaults(self):
        from eval.chat_benchmarks.LiveCodeBench.eval_instruct import LiveCodeBenchBenchmark

        self.assertEqual(LiveCodeBenchBenchmark(version="v2").version_tag, "v2")
        self.assertEqual(LiveCodeBenchBenchmark(version="release_v5").version_tag, "release_v5")
        self.assertEqual(LiveCodeBenchBenchmark(version=6).version_tag, "v6")
        self.assertEqual(LiveCodeBenchBenchmark(version="v5_v6").version_tag, "v5_v6")
        self.assertEqual(LiveCodeBenchBenchmark(version="release_v5_v6").version_tag, "release_v5_v6")
        self.assertEqual(LiveCodeBenchBenchmark(version="v2").n_repeat, 6)
        self.assertEqual(LiveCodeBenchBenchmark(version="v6").n_repeat, 3)
        self.assertEqual(LiveCodeBenchBenchmark(version="v5_v6").n_repeat, 3)

    def test_livecodebench_load_questions_uses_version_tag(self):
        from eval.chat_benchmarks.LiveCodeBench import eval_instruct as lcb_module

        benchmark = lcb_module.LiveCodeBenchBenchmark(version="v6")

        with mock.patch.object(lcb_module, "load_dataset", return_value=FakeDataset()) as load_dataset_mock:
            with mock.patch.object(lcb_module, "concatenate_datasets", side_effect=lambda shards: shards[0]):
                benchmark.load_questions()

        load_dataset_mock.assert_called_once_with(
            "livecodebench/code_generation_lite",
            version_tag="v6",
            split="test",
            trust_remote_code=True,
            cache_dir=lcb_module.HF_HUB_CACHE,
        )

    def test_livecodebench_load_questions_supports_delta_version_tags(self):
        from eval.chat_benchmarks.LiveCodeBench import eval_instruct as lcb_module

        benchmark = lcb_module.LiveCodeBenchBenchmark(version="v5_v6")

        with mock.patch.object(lcb_module, "load_dataset", return_value=FakeDataset()) as load_dataset_mock:
            with mock.patch.object(lcb_module, "concatenate_datasets", side_effect=lambda shards: shards[0]):
                benchmark.load_questions()

        load_dataset_mock.assert_called_once_with(
            "livecodebench/code_generation_lite",
            version_tag="v5_v6",
            split="test",
            trust_remote_code=True,
            cache_dir=lcb_module.HF_HUB_CACHE,
        )

    def test_livecodebench_single_example_includes_public_test_results(self):
        from eval.chat_benchmarks.LiveCodeBench import eval_instruct as lcb_module

        benchmark = lcb_module.LiveCodeBenchBenchmark(version="v6")
        example = {
            "difficulty": "easy",
            "model_answer": ["print(2)"],
            "public_test_cases": '[{"input": "1\\n", "output": "2", "testtype": "stdin"}]',
            "is_stdin": True,
        }

        with mock.patch.object(benchmark, "check_correctness", return_value=False):
            with mock.patch.object(
                lcb_module,
                "lcb_run_test_cases",
                return_value=[(True, "public test passed", "2", 0.01)],
            ):
                result = benchmark.evaluate_single_example(example)

        self.assertFalse(result["correctness"])
        self.assertEqual(result["reason"], "Code is incorrect.")
        self.assertEqual(
            result["public_test_results"],
            [
                {
                    "test_index": 0,
                    "test_case": {"input": "1\n", "output": "2", "testtype": "stdin"},
                    "passed": True,
                    "details": "public test passed",
                    "output": "2",
                    "time_elapsed": 0.01,
                }
            ],
        )

    def test_task_manager_forwards_livecodebench_version(self):
        from eval.task import TaskManager

        task_manager = TaskManager(task_list=["LiveCodeBench"], version="v5_v6", debug=True)
        benchmark = task_manager.get_benchmark("LiveCodeBench")

        self.assertIsNotNone(benchmark)
        self.assertEqual(benchmark.version_tag, "v5_v6")
        self.assertEqual(benchmark.n_repeat, 3)

    def test_livecodebench_legacy_wrappers_are_thin_configs(self):
        from eval.chat_benchmarks.LiveCodeBenchv5.eval_instruct import LiveCodeBenchV5Benchmark
        from eval.chat_benchmarks.LiveCodeBenchv5_official.eval_instruct import LiveCodeBenchV5OfficialBenchmark

        legacy_v5 = LiveCodeBenchV5Benchmark()
        official_v5 = LiveCodeBenchV5OfficialBenchmark()

        self.assertEqual(legacy_v5.version_tag, "v5")
        self.assertEqual(legacy_v5.dataset_repo, "mlfoundations-dev/LCBv5-v2")
        self.assertIsNone(legacy_v5.contest_months)

        self.assertEqual(official_v5.version_tag, "v5")
        self.assertEqual(official_v5.dataset_repo, "livecodebench/code_generation_lite")
        self.assertEqual(
            official_v5.contest_months,
            {"2024-08", "2024-09", "2024-10", "2024-11", "2024-12", "2025-01"},
        )


if __name__ == "__main__":
    unittest.main()
