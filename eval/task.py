import importlib.util
import inspect
import logging
import os
import random
import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Type, Union

import lm_eval.models as lm_eval_models
import numpy as np
import torch
import torch.distributed as dist
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM


@dataclass
class TaskInstance:
    """Generation-ready benchmark sample with a raw-output evaluator."""

    id: str
    doc: Any
    request_type: str
    prompt: Any
    generation_kwargs: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    _evaluate_fn: Optional[Callable[["TaskInstance", str], Dict[str, Any]]] = field(default=None, repr=False)

    def evaluate(self, raw_output: str) -> Dict[str, Any]:
        if self._evaluate_fn is None:
            raise NotImplementedError("No evaluator is attached to this task instance.")
        return self._evaluate_fn(self, raw_output)


class CaptureLM:
    """Small LM stand-in that records generation requests without generating."""

    rank = 0
    world_size = 1

    def __init__(self, wrapped_model: Optional[LM] = None):
        self.wrapped_model = wrapped_model
        self.model_args = getattr(wrapped_model, "model_args", {"model": "capture"})
        self.model = getattr(wrapped_model, "model", "capture")
        self.captured_instances: List[Instance] = []

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> Union[List[Dict[str, str]], str]:
        if self.wrapped_model is not None:
            return self.wrapped_model.apply_chat_template(messages)
        return messages

    def generate_until(self, requests: List[Instance]) -> List[str]:
        self.captured_instances.extend(requests)
        return ["" for _ in requests]


class BaseBenchmark(ABC):
    """Abstract base class for implementing LLM evaluation benchmarks."""

    def __init__(self, logger: Optional[logging.Logger] = None, system_instruction: Optional[str] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.system_instruction = system_instruction

    def _normalize_model_args(self, model: LM, instances: List[Instance]) -> List[Instance]:
        openai_completions = getattr(lm_eval_models, "openai_completions", None)
        vllm_causallms = getattr(lm_eval_models, "vllm_causallms", None)

        for instance in instances:
            seeds = None
            if "seed" in instance.args[1]:
                seeds = instance.args[1]["seed"]

                random.seed(seeds[0])
                np.random.seed(seeds[1])
                torch.manual_seed(seeds[2])

                is_openai_model = openai_completions is not None and (
                    isinstance(model, openai_completions.OpenAIChatCompletion)
                    or isinstance(model, openai_completions.OpenAICompletionsAPI)
                )
                is_vllm_model = vllm_causallms is not None and isinstance(model, vllm_causallms.VLLM)

                if is_openai_model:
                    instance.args[1]["seed"] = seeds[0] if "seed" in instance.args[1] else None
                elif is_vllm_model or "UploadInstancesToHF" in model.__class__.__name__:
                    instance.args[1]["seed"] = seeds[0] if "seed" in instance.args[1] else None
                else:  # Huggingface does not support seed
                    _ = instance.args[1].pop("seed") if "seed" in instance.args[1] else None
            if "max_new_tokens" in instance.args[1]:
                max_new_tokens = instance.args[1].pop("max_new_tokens")
                is_openai_model = openai_completions is not None and (
                    isinstance(model, openai_completions.OpenAIChatCompletion)
                    or isinstance(model, openai_completions.OpenAICompletionsAPI)
                )
                is_vllm_model = vllm_causallms is not None and isinstance(model, vllm_causallms.VLLM)

                if is_openai_model:
                    instance.args[1]["max_tokens"] = max_new_tokens
                    if "4o" in model.model:
                        instance.args[1]["max_tokens"] = min(max_new_tokens, 16384)
                elif is_vllm_model:
                    instance.args[1]["max_gen_toks"] = max_new_tokens
                else:  # Huggingface
                    instance.args[1]["max_new_tokens"] = max_new_tokens
        return instances

    def _prepare_messages(
        self, messages: List[Dict[str, str]], model: Optional[LM] = None
    ) -> Union[List[Dict[str, str]], str]:
        """Prepare messages with system instruction if available and apply chat template if model is provided.

        Args:
            messages: List of message dictionaries
            model: Optional language model instance for applying chat template

        Returns:
            If model is provided, returns the templated string. Otherwise returns the prepared message list.
        """
        if self.system_instruction:
            messages.insert(0, {"role": "system", "content": self.system_instruction})

        if model is not None:
            return model.apply_chat_template(messages)

        return messages

    def compute(self, model: LM, inputs: List[Instance], do_slice: bool = True) -> List[str]:
        normalize_model = model.wrapped_model if isinstance(model, CaptureLM) and model.wrapped_model else model
        inputs = self._normalize_model_args(normalize_model, inputs)

        # Add task_name to each instance
        task_name = self.__class__.__name__.replace("Benchmark", "")
        for instance in inputs:
            instance.task_name = task_name

        if model.world_size > 1 and do_slice:
            prompts = list(islice(inputs, model.rank, len(inputs), model.world_size))
        else:
            prompts = inputs

        results = model.generate_until(prompts)
        if model.world_size > 1:
            all_results = [None for _ in range(model.world_size)]

            dist.all_gather_object(all_results, results)

            # Merge results from all ranks
            length = sum(len(res) for res in all_results if res is not None)
            merged = [None] * length
            for rank, sub_results in enumerate(all_results):
                if sub_results is not None:
                    for i, item in enumerate(sub_results):
                        merged[i * model.world_size + rank] = item
            return merged
        else:
            return results

    def task_instances(self, model: Optional[LM] = None) -> List[TaskInstance]:
        """
        Extract generation-ready task instances from this benchmark.

        The default implementation reuses the benchmark's existing
        generate_responses method with a capture-only model, so benchmark CLI
        behavior does not need to change.
        """
        capture_lm = CaptureLM(model)
        self.generate_responses(capture_lm)
        return [self._task_instance_from_lm_instance(instance) for instance in capture_lm.captured_instances]

    def _task_instance_from_lm_instance(self, instance: Instance) -> TaskInstance:
        prompt, generation_kwargs = instance.args
        doc = getattr(instance, "doc", None)
        idx = getattr(instance, "idx", None)
        if idx is None:
            idx = getattr(instance, "request_index", len(getattr(self, "_task_instances", [])))

        instance_id = None
        if isinstance(doc, dict):
            instance_id = doc.get("id") or doc.get("task_id") or doc.get("question_id")
        if instance_id is None:
            instance_id = str(idx)

        metadata = {
            "idx": idx,
            "task_name": getattr(instance, "task_name", self.__class__.__name__.replace("Benchmark", "")),
        }

        return TaskInstance(
            id=str(instance_id),
            doc=deepcopy(doc),
            request_type=getattr(instance, "request_type", "generate_until"),
            prompt=deepcopy(prompt),
            generation_kwargs=deepcopy(dict(generation_kwargs)),
            metadata=metadata,
            _evaluate_fn=self.evaluate_task_instance,
        )

    def evaluate_task_instance(self, task_instance: TaskInstance, raw_output: str) -> Dict[str, Any]:
        """
        Best-effort raw-output evaluation for one extracted task instance.

        Benchmarks with custom file-based or batch-only evaluation can override
        this method. For the common in-memory examples shape, this injects the
        raw output into a copied doc and delegates to evaluate_responses.
        """
        if task_instance.doc is None:
            return {
                "supported": False,
                "raw_output": raw_output,
                "reason": "Task instance has no source doc to evaluate.",
            }

        doc = deepcopy(task_instance.doc)
        if isinstance(doc, dict):
            doc["model_output"] = raw_output
            if hasattr(self, "extract_answer") and callable(self.extract_answer):
                doc["model_answer"] = self.extract_answer(raw_output)
            else:
                for cls in self.__class__.mro():
                    benchmark_module = sys.modules.get(cls.__module__)
                    has_code = getattr(benchmark_module, "has_code", None)
                    if callable(has_code):
                        doc["model_answer"] = has_code(raw_output)
                        break
            doc["model_outputs"] = [raw_output]
            if "model_answer" in doc:
                doc["model_answers"] = [doc["model_answer"]]

        try:
            result = self.evaluate_responses({"examples": [doc]})
        except Exception as exc:
            return {
                "supported": False,
                "raw_output": raw_output,
                "reason": f"Benchmark evaluator could not score one sample: {exc}",
            }

        return {
            "supported": True,
            "raw_output": raw_output,
            "result": result,
        }

    @abstractmethod
    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """Generate responses from the model for the benchmark tasks."""
        pass

    @abstractmethod
    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the model's responses according to the benchmark's metrics."""
        pass

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """Run the complete benchmark evaluation pipeline."""
        print(f"Running {self.__class__.__name__} benchmark")
        generation_results = self.generate_responses(model)
        evaluation_results = self.evaluate_responses(generation_results)
        return evaluation_results


class TaskManager:
    """
    Enhanced task manager that dynamically loads and manages benchmarks.
    Provides a unified interface for both class-based benchmarks and legacy tasks.
    """

    def __init__(
        self, benchmarks_dir: str = "chat_benchmarks", task_list: Optional[List[str]] = None, **benchmark_kwargs
    ):
        self.logger = logging.getLogger("TaskManager")
        self.tasks: Dict[str, Any] = {}
        self.benchmark_instances: Dict[str, BaseBenchmark] = {}
        self.benchmark_kwargs = benchmark_kwargs
        self.task_list = task_list
        self.list_of_tasks_that_require_annotator_model = []

        # Load benchmarks from directory
        self._load_benchmarks(benchmarks_dir)

    def _load_benchmarks(self, benchmarks_dir: str):
        """Dynamically load benchmarks from the specified directory."""
        current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), benchmarks_dir)

        # Check if OpenAI API key is available
        has_openai_key = os.getenv("OPENAI_API_KEY") is not None
        if not has_openai_key:
            self.logger.warning("OPENAI_API_KEY not set. Tasks requiring OpenAI will be skipped.")

        # Temporarily set the API key to an empty string to prevent NoneType errors
        if not has_openai_key:
            os.environ["OPENAI_API_KEY"] = ""  # Empty string instead of None

        for item in os.listdir(current_dir):
            # Skip loading if task_list is provided and this item is not in it
            if self.task_list is not None and item not in self.task_list:
                continue

            item_path = os.path.join(current_dir, item)
            if not os.path.isdir(item_path) or item.startswith("__"):
                continue

            eval_path = os.path.join(item_path, "eval_instruct.py")
            if not os.path.exists(eval_path):
                self.logger.warning(f"eval_instruct.py not found in {item}")
                continue

            try:
                # Import the module
                sys.path.insert(0, item_path)
                spec = importlib.util.spec_from_file_location(f"eval.{benchmarks_dir}.{item}.eval_instruct", eval_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.path.pop(0)

                # Find benchmark class
                benchmark_classes = [
                    cls
                    for _, cls in inspect.getmembers(module, inspect.isclass)
                    if (
                        issubclass(cls, BaseBenchmark)
                        and cls != BaseBenchmark
                        and cls.__module__.replace(".", "/") in eval_path
                    )
                ]

                if not benchmark_classes:
                    self.logger.warning(f"No BaseBenchmark subclass found in {item}")
                    continue

                if len(benchmark_classes) > 1:
                    self.logger.warning(f"Multiple benchmark classes found in {item}, using first one")

                benchmark_class = benchmark_classes[0]

                # Check if this benchmark requires OpenAI as annotator model
                requires_annotator = "annotator_model" in inspect.signature(benchmark_class.__init__).parameters

                # Check if the benchmark explicitly requires OpenAI for annotation
                requires_openai = (
                    hasattr(benchmark_class, "REQUIRES_OPENAI_ANNOTATOR") and benchmark_class.REQUIRES_OPENAI_ANNOTATOR
                )

                if requires_annotator:
                    self.list_of_tasks_that_require_annotator_model.append(item)

                if not has_openai_key and requires_openai:
                    self.logger.warning(
                        f"Not loading {item} benchmark as it requires OpenAI as annotator model but OPENAI_API_KEY is not set"
                    )
                    continue

                self._register_benchmark(item, benchmark_class)

            except Exception as e:
                self.logger.error(f"Error loading benchmark from {item}: {str(e)}")
                continue

        # Clean up temporary environment variable if we set it
        if not has_openai_key and "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"] == "":
            del os.environ["OPENAI_API_KEY"]

    def _register_benchmark(self, name: str, benchmark_class: Type[BaseBenchmark]):
        """Register a benchmark class and create its instance."""
        try:
            init_params = inspect.signature(benchmark_class.__init__).parameters
            valid_kwargs = {}

            # Only pass kwargs that the benchmark's __init__ accepts
            # Filter out None values to let benchmarks use their default values
            for param_name, param in init_params.items():
                if param_name in self.benchmark_kwargs:
                    value = self.benchmark_kwargs[param_name]
                    # Only pass the argument if it's not None, so benchmarks can use defaults
                    if value is not None:
                        valid_kwargs[param_name] = value
                        self.logger.debug(f"Passing {param_name}={value} to {name} benchmark")

            # Ensure system_instruction is passed if available and not None
            if (
                "system_instruction" in self.benchmark_kwargs
                and self.benchmark_kwargs["system_instruction"] is not None
            ):
                valid_kwargs["system_instruction"] = self.benchmark_kwargs["system_instruction"]

            instance = benchmark_class(**valid_kwargs)

            self.tasks[name] = benchmark_class
            self.benchmark_instances[name] = instance

            self.logger.debug(f"Successfully registered benchmark: {name}")

        except Exception as e:
            self.logger.error(f"Error registering benchmark {name}: {str(e)}")

    def get_list_generate_responses(self, task_list: List[str]) -> List[Callable]:
        """Get list of generate_responses methods for given tasks."""
        methods = []
        for task in task_list:
            if task in self.benchmark_instances:
                methods.append(self.benchmark_instances[task].generate_responses)
            else:
                self.logger.warning(f"Task not found: {task}")
        return methods

    def get_list_evaluates(self, task_list: List[str]) -> List[Callable]:
        """Get list of evaluate_responses methods for given tasks."""
        methods = []
        for task in task_list:
            if task in self.benchmark_instances:
                methods.append(self.benchmark_instances[task].evaluate_responses)
            else:
                self.logger.warning(f"Task not found: {task}")
        return methods

    @property
    def available_tasks(self) -> List[str]:
        """Get list of all available tasks."""
        return list(self.tasks.keys())

    def get_benchmark(self, name: str) -> Optional[BaseBenchmark]:
        """Get a benchmark instance by name."""
        return self.benchmark_instances.get(name)

    def is_valid_task(self, task_name: str) -> bool:
        """Check if a task name is valid."""
        return task_name in self.tasks

    def requires_annotator_model(self, task_name: str) -> bool:
        """
        Check if a task requires an annotator model by inspecting its __init__ signature.

        Args:
            task_name: The name of the task to check

        Returns:
            bool: True if the task's __init__ has an annotator_model parameter, False otherwise
        """
        if task_name in self.list_of_tasks_that_require_annotator_model:
            return True
        if task_name not in self.tasks:
            self.logger.warning(f"Task not found: {task_name}")
            return False

        task_cls = self.tasks[task_name]

        # Get the signature of the task's __init__ method
        init_params = inspect.signature(task_cls.__init__).parameters

        # Check if 'annotator_model' is in the parameters
        return "annotator_model" in init_params


def evaluate(
    lm: LM, task_manager: TaskManager, task_list: List[str], verbosity: str = "INFO", **eval_kwargs
) -> Dict[str, Dict]:
    """
    Evaluate the language model on the given tasks.

    Args:
        lm: The language model to evaluate
        task_manager: Task manager containing the benchmarks
        task_list: List of task names to evaluate
        verbosity: Logging verbosity level
        **eval_kwargs: Additional kwargs for evaluation

    Returns:
        Dictionary containing evaluation results for each task
    """
    logger = logging.getLogger("evaluate")
    logger.setLevel(getattr(logging, verbosity))

    results = {"results": {}}

    # Validate tasks
    valid_tasks = [t for t in task_list if task_manager.is_valid_task(t)]
    if len(valid_tasks) != len(task_list):
        invalid_tasks = set(task_list) - set(valid_tasks)
        logger.warning(f"Skipping invalid tasks: {invalid_tasks}")

    if not valid_tasks:
        logger.error("No valid tasks to evaluate")
        return results

    # Run evaluations
    for task_name in valid_tasks:
        try:
            benchmark = task_manager.get_benchmark(task_name)
            if benchmark:
                logger.info(f"Evaluating {task_name}")
                results["results"][task_name] = benchmark.run_benchmark(lm)
        except Exception as e:
            logger.error(f"Error evaluating {task_name}: {str(e)}")
            results["results"][task_name] = {"error": str(e)}

    return results


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Initialize task manager
    task_manager = TaskManager()

    # Print available tasks
    print("Available tasks:", task_manager.available_tasks)
