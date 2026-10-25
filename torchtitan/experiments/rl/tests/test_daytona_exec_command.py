"""``_build_exec_command`` run through a local bash, as the Daytona session runs it."""

import subprocess

import pytest

from torchtitan.experiments.rl.harness.sandbox import daytona


def _run(cmd: str, tmp_path, stdin: str = "") -> subprocess.CompletedProcess:
    script = tmp_path / "wrapper.sh"
    script.write_text(daytona._build_exec_command(cmd, user="root", env=None, timeout=10))
    return subprocess.run(
        ["bash", str(script)], input=stdin, capture_output=True, text=True, check=False
    )


def test_errors_read_like_a_plain_shell_not_the_staging_path(tmp_path):
    proc = _run("no_such_cmd_zz", tmp_path)
    assert proc.returncode == 127
    assert proc.stderr.startswith("bash: line 1: no_such_cmd_zz: command not found")
    assert daytona._EXEC_STAGING_DIR not in proc.stderr


def test_stdin_exit_code_and_heredoc_survive(tmp_path):
    proc = _run("read x; cat <<'EOF'\n$x stays literal\nEOF\necho got=$x; exit 3", tmp_path, "hi\n")
    assert (proc.returncode, proc.stdout) == (3, "$x stays literal\ngot=hi\n")


@pytest.mark.parametrize("pad", [100, daytona._EXEC_INLINE_SCRIPT_MAX + 10])
def test_a_script_past_the_inline_cap_still_runs_from_the_file(tmp_path, pad):
    proc = _run(f"echo {'y' * pad} | wc -c", tmp_path)
    assert (proc.returncode, proc.stdout.strip()) == (0, str(pad + 1))
