import re
import shutil
import subprocess
import sys
from pathlib import Path

_DEMO_REPO = Path(__file__).resolve().parents[1] / "examples" / "demo-repo"


def test_demo_repo__clean_copy_has_exactly_one_seed_failure(tmp_path: Path) -> None:
    workspace = shutil.copytree(
        _DEMO_REPO,
        tmp_path / "demo-repo",
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.py[cod]"),
    )

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=workspace,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    output = f"{completed.stdout}\n{completed.stderr}"
    failed_counts = [int(count) for count in re.findall(r"(?<!\d)(\d+) failed\b", output)]
    passed_counts = [int(count) for count in re.findall(r"(?<!\d)(\d+) passed\b", output)]
    error_counts = [int(count) for count in re.findall(r"(?<!\d)(\d+) errors?\b", output)]

    assert completed.returncode == 1, output
    assert failed_counts and failed_counts[-1] == 1, output
    assert passed_counts and passed_counts[-1] == 3, output
    assert not error_counts, output
    assert "FAILED tests/test_ops.py::test_divide" in output, output
