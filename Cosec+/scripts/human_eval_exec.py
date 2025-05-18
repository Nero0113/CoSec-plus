import argparse
import os
import sys

sys.path.append('../')
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Optional

import yaml
from tqdm import tqdm

from utils import Problem


# program: str => Result
CACHE = dict()
CACHE_LOCK = Lock()

def get_test_results_yaml_path(problem_yaml_path: Path) -> Path:
    return problem_yaml_path.parent / (problem_yaml_path.stem + ".results.yaml")

def cache_get(program: str) -> Optional[dict]:
    if program in CACHE:
        result = CACHE[program]
        return result
    else:
        return None

def cache_set(program: str, result: dict):
    if program in CACHE:
        print("Setting already-existing cache")
    CACHE[program] = result

def eval_script_python(path: Path):

    output = None
    try:
        # Assumes exit-code 0 is all okay
        output = subprocess.run(
            ["python", str(path)], encoding="utf-8", capture_output=True, timeout=5
        )
        returncode = -1
        if output.returncode == 0:
            status = "OK"
            returncode = output.returncode
        elif "SyntaxError" in output.stderr:
            status = "SyntaxError"
            returncode = output.returncode
        else:
            status = "Exception"

    except subprocess.TimeoutExpired as exc:
        status = "Timeout"
        returncode = -1
        output = exc

    return {
        "status": status,
        "exit_code": returncode,
        "stdout": str(output.stdout),
        "stderr": str(output.stderr),
    }

def eval_string_script(language, program):

    eval_script, file_ext = eval_script_python, '.py'
    '''
    tempfile：这是 Python 标准库中的模块，用于处理临时文件和目录的创建和管理。
    NamedTemporaryFile：这是 tempfile 模块提供的一个类，用于创建具有唯一名称的临时文件。
    suffix：这是 NamedTemporaryFile 类的一个参数，用于指定临时文件的后缀名。
    delete：这是 NamedTemporaryFile 类的一个参数，用于指定临时文件是否在关闭时删除。
    '''
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=True) as f:
        f.write(program.encode("utf-8"))
        f.flush()
        result = eval_script(Path(f.name))
        # 只保存运行程序的前 2K 输出。任何后续输出都很可能是超长的堆栈跟踪或一长串打印。
        if type(result["stdout"]) == bytes:
            result["stdout"] = result["stdout"].decode("utf-8", errors="ignore")
        if result["stdout"] is None:
            result["stdout"] = ""
        if result["stderr"] is None:
            result["stderr"] = ""
        if type(result["stderr"]) == bytes:
            result["stderr"] = result["stderr"].decode("utf-8", errors="ignore")

        assert type(result["stdout"]) == str
        assert type(result["stderr"]) == str

        return {
            "program": program,
            "stdout": result['stdout'].replace("!!int", "")[:2048],
            "stderr": result['stderr'][:2048],
            "exit_code": result['exit_code'],
            "status": result['status']
        }





def cached_eval_script(problem, index) -> dict:
    program = problem.prompt + problem.completions[index] + '\n' + problem.tests
    CACHE_LOCK.acquire(True)
    cached = cache_get(program)
    if cached is not None:
        CACHE_LOCK.release()
        return cached
    else:
        result_yaml = dict()
        cache_set(program, result_yaml)
        CACHE_LOCK.release()
        result_dict = eval_string_script(problem.language, program)
        for k in result_dict.keys():
            result_yaml[k] = result_dict[k]
            result_yaml["timestamp"] = int(time.time())
        return result_yaml

class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

def evaluate_problem(problem_yaml_path: Path, max_workers: int):
    with open(problem_yaml_path) as f:
        problem = Problem.load(f)

        # Do not create a blank .results.yaml file if there are no completions ready.
        if len(problem.completions) == 0:
            return

        test_results_path = get_test_results_yaml_path(problem_yaml_path)

        if not test_results_path.exists():
            test_results = {
                "name": problem.name,
                "language": problem.language,
                "results": [],
            }
        else:
            with test_results_path.open() as f:
                test_results = yaml.safe_load(f)

        num_problems = len(problem.completions)

        if len(test_results["results"]) == num_problems:
            return

        elif len(test_results["results"]) > num_problems:
            print(f"ERROR more results than completions for {problem_yaml_path}")
            return

        min_problem = len(test_results["results"])

        # In case we have previously computed results, warm the cache with them
        for already_computed in test_results["results"]:
            CACHE[already_computed["program"]] = already_computed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for j in executor.map(lambda index: cached_eval_script(problem, index), range(min_problem, num_problems)):
                test_results["results"].append(j)

                with test_results_path.open("w") as f:
                    f.write(yaml.dump(test_results, Dumper=NoAliasDumper))



if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--output_name", type=str, default='human-eval-Qwen2.5-Coder-14B-seconly')
    args.add_argument("--max_workers", type=int, default=50)
    args.add_argument('--output_dir', type=str, default='/path/to/CoSec2/experiments/')
    args.add_argument('--eval_type', type=str, default='human_eval')
    args = args.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.eval_type, args.output_name)

    files = [p for p in Path(args.output_dir).glob("*.yaml") if not p.name.endswith(".results.yaml")]

    for file in tqdm(files):
        evaluate_problem(file, args.max_workers)
