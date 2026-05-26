"""
Code from https://github.com/NovaSky-AI/SkyThought/blob/main/skythought/tools/util/livecodebench/testing_util.py
"""

import ast
import base64
import builtins
import copy
import contextlib
import faulthandler
import io
import json
import multiprocessing
import pickle
import sys
import time
import zlib
from typing import Callable, Dict, Optional

import scipy.stats as stats


def _get_mp_context():
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return multiprocessing.get_context()


def _safe_text(value, max_length=1000):
    text = str(value)
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def _safe_test_result(result):
    passed, details, output, time_elapsed = result
    return passed, _safe_text(details), _safe_text(output), time_elapsed


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    if callable(getattr(os, "putenv", None)):
        os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    # __builtins__["help"] = None   # this line is commented out as it results into error

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def has_test_type(tests, type):  ## helper to select specific type of problems
    """
    Check if any test in the test list has 'testtype' set to 'type'.
    """
    test_list = json.loads(tests)
    for test in test_list:
        if test.get("testtype") == type:
            return True
    return False


def translate_private_test_cases(encoded_data):
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    return json.loads(original_data)


def map_to_example(row):
    return {
        "prompt": row["question_content"],
        "test": row["private_test_cases"],
        "entry_point": row["starter_code"],
        "task_id": row["question_id"],
        "is_stdin": has_test_type(row["public_test_cases"], "stdin"),
        "public_test_cases": row["public_test_cases"],
        "difficulty": row["difficulty"],
    }


def post_process_code(code):
    code = code.split("</code>")[0]
    code = code.replace("```python", "")
    code = code.split("```")[0]
    code = code.replace("<code>", "")
    return code


def prepare_test_input_output_std(test_case):
    test_input = test_case["input"]
    test_output = test_case["output"].strip()
    if test_output.endswith("-"):
        test_output = test_output[: test_output.rfind("-")].rstrip()  # Remove '-' if present and trailing
    return test_input, test_output


def run_test_func(completion, is_extracted, test_input, test_output):
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(completion, namespace)
        if not is_extracted:
            func_name = completion.split("(")[0].split()[-1]
            if isinstance(test_input, dict):
                result_output = namespace[func_name](**test_input)
            else:
                result_output = namespace[func_name](test_input)
        else:
            func_name = completion.split("(")[0].split()[-1]
            result_output = namespace[func_name](*test_input)

    return result_output == test_output, result_output


def run_test_std(completion, test_input, test_output):
    with io.StringIO() as output, contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
        sys.stdin = io.StringIO(test_input)
        try:
            exec(f'__name__ = "__main__"\n{completion}' if '__name__ == "__main__"' in completion else completion, {})
            return output.getvalue().strip() == test_output, output.getvalue().strip()
        finally:
            sys.stdin = sys.__stdin__


def prepare_test_input_output_functional(test_case, is_extracted):
    if not is_extracted:
        # Extract input and expected output from JSON directly
        test_input = test_case["input"]
        test_output = test_case["output"]
        return test_input, test_output
    else:
        # Robustly process complex inputs
        input_str = test_case["input"]
        expected_output = test_case["output"].strip()
        inputs = []

        if "=" in input_str:
            parts = input_str.split(",") if "," in input_str else [input_str]
            for part in parts:
                key, value = map(str.strip, part.split("="))
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        value = value.strip('"')
                inputs.append(value)
        else:
            for line in input_str.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.startswith('"') and line.endswith('"'):
                    inputs.append(line.strip('"'))
                    continue
                if line.startswith("[") and line.endswith("]"):
                    inputs.append(json.loads(line))
                    continue
                try:
                    inputs.append(int(line))
                except ValueError:
                    try:
                        inputs.append(float(line))
                    except ValueError:
                        inputs.append(line)

        try:
            expected_output = json.loads(expected_output)
        except json.JSONDecodeError:
            expected_output = expected_output.strip()
        return inputs, expected_output


def run_tests_for_one_example(test_cases, completion, result_list, is_extracted):
    time_elapsed = float("inf")
    test_type = test_cases[0]["testtype"]
    reliability_guard()
    for i, test_case in enumerate(test_cases):
        output_error = ""
        output_value = ""
        try:
            time_start = time.time()
            if test_type == "functional":
                test_input, test_output = prepare_test_input_output_functional(test_case, is_extracted)
                passed, output_value = run_test_func(
                    completion, is_extracted, copy.deepcopy(test_input), copy.deepcopy(test_output)
                )
            else:
                test_input, test_output = prepare_test_input_output_std(test_case)
                passed, output_value = run_test_std(completion, copy.deepcopy(test_input), copy.deepcopy(test_output))
            time_elapsed = time.time() - time_start
            if not passed:
                output_error = (
                    f"For test input: {test_input}. Expected output is: {test_output}, but got: {output_value}."
                )

        except Exception as e:
            passed = False
            output_error = f"For test input: {test_input}. Expected output is: {test_output}, but got error: {e}."
            output_value = f"Error: {e}."
        if output_error == "":
            output_error = f"For test input: {test_input}. Expected output is: {test_output}, your solution correctly passes this test with output {output_value}."
        result_list.append(_safe_test_result((passed, output_error, output_value, time_elapsed)))
        if not passed:
            return


def lcb_run(problem, completion, timeout, is_extracted):
    test_cases = problem["test"]
    return lcb_run_test_cases(test_cases, completion, timeout, is_extracted)


def lcb_run_test_sets(private_test_cases, public_test_cases, completion, timeout, is_extracted):
    hard_timeout = (timeout + 1) * max(len(private_test_cases) + len(public_test_cases), 1) + 5
    result = _run_in_subprocess(
        _run_test_sets_worker,
        (private_test_cases, public_test_cases, completion, is_extracted),
        hard_timeout,
        {"private": [], "public": []},
    )
    return {
        "private": _pad_timed_out_results(result.get("private", []), len(private_test_cases)),
        "public": _pad_timed_out_results(result.get("public", []), len(public_test_cases)),
    }


def lcb_run_test_cases(test_cases, completion, timeout, is_extracted):
    hard_timeout = (timeout + 1) * len(test_cases) + 5
    result = _run_in_subprocess(
        _run_tests_worker,
        (test_cases, completion, is_extracted),
        hard_timeout,
        [],
    )
    return _pad_timed_out_results(result, len(test_cases))


def _run_in_subprocess(target, args, timeout, default):
    context = _get_mp_context()
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(*args, child_conn))
    process.start()
    child_conn.close()

    deadline = time.time() + timeout
    while process.is_alive() and time.time() < deadline:
        if parent_conn.poll(0.1):
            return _finish_process(process, parent_conn, default)

    process.join(0)
    if process.is_alive():
        process.kill()
        process.join()
    try:
        if parent_conn.poll():
            return parent_conn.recv()
    except (EOFError, OSError):
        pass
    finally:
        parent_conn.close()
    return default


def _finish_process(process, parent_conn, default):
    try:
        result = parent_conn.recv()
    except (EOFError, OSError):
        result = default
    process.join(1)
    if process.is_alive():
        process.kill()
        process.join()
    parent_conn.close()
    return result


def _run_tests_worker(test_cases, completion, is_extracted, conn):
    result = []
    try:
        run_tests_for_one_example(test_cases, completion, result, is_extracted)
        conn.send(result)
    except BaseException as exc:
        conn.send([(False, f"Evaluation error: {_safe_text(exc)}.", f"Error: {_safe_text(exc)}", float("inf"))])
    finally:
        conn.close()


def _run_test_sets_worker(private_test_cases, public_test_cases, completion, is_extracted, conn):
    public_result = []
    private_result = []
    try:
        if public_test_cases:
            run_tests_for_one_example(public_test_cases, completion, public_result, is_extracted)
        if private_test_cases:
            run_tests_for_one_example(private_test_cases, completion, private_result, is_extracted)
        conn.send({"public": public_result, "private": private_result})
    except BaseException as exc:
        error = (False, f"Evaluation error: {_safe_text(exc)}.", f"Error: {_safe_text(exc)}", float("inf"))
        conn.send({"public": public_result, "private": private_result or [error]})
    finally:
        conn.close()


def _pad_timed_out_results(result, num_test_cases):
    result = list(result)
    for _ in range(num_test_cases - len(result)):
        result.append((False, "Time out!.", "Error: Time out!", float("inf")))
    return result
