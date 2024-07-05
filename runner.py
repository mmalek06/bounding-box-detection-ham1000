import asyncio
import os
import subprocess
import sys
from asyncio import WindowsSelectorEventLoopPolicy
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm


def should_exit(exit_file: str) -> bool:
    path = os.path.join(os.path.expanduser('~'), f'.{exit_file}')
    return os.path.isfile(path)


def _get_runs_file_path(run_file: str) -> str:
    return os.path.join(os.path.expanduser('~'), f'.{run_file}')


def get_run_number(run_file: str) -> int:
    run_files_path = _get_runs_file_path(run_file)
    run_file = Path(run_files_path)

    run_file.parent.mkdir(exist_ok=True, parents=True)
    run_file.touch(exist_ok=True)

    text = run_file.read_text()

    if len(text) > 0:
        return int(text)

    return 1


def get_runs_data(allowed_files: set[str]) -> dict[str, int]:
    files = os.listdir(".")
    run_data = defaultdict(lambda: 1)

    for file in files:
        if file not in allowed_files:
            continue

        file_path = os.path.join(".", file)

        if os.path.isdir(file_path):
            continue

        file_base_name = file.split('.')[0]
        run_file_name = f'{file_base_name}'
        run_number = get_run_number(run_file_name)
        run_data[file_path] = run_number

    return run_data


load_dotenv()

RUN_TIMES = int(os.getenv("RUN_TIMES"))
EXIT_FILE = os.getenv("EXIT_FILE")
NOTEBOOKS = set(os.getenv("NOTEBOOKS").split(","))
runs_data = get_runs_data(NOTEBOOKS)
total_runs = len(runs_data) * RUN_TIMES

with tqdm(total=total_runs, desc="Processing Notebooks") as pbar:
    for notebook_path, run in runs_data.items():
        if run >= RUN_TIMES + 1:
            continue

        for _ in range(RUN_TIMES - run + 1):
            if should_exit(EXIT_FILE):
                print('Exit file encountered, aborting...')
                sys.exit(0)

            conda_activate_command = "conda run -n bounding_box_detection_ham10000_torch && "
            command = (f"{conda_activate_command}jupyter nbconvert --execute --to notebook --inplace "
                       f"--ExecutePreprocessor.kernel_name=python3 {notebook_path}")
            os.environ["RUN_NUMBER"] = str(run)
            os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = str(1)

            subprocess.run(command, shell=True)

            pbar.update(1)
