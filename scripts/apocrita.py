#!/usr/bin/env python3
"""Execute a command inside an existing Apocrita OOD allocation via SSH/srun."""
import argparse
import shlex
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=int)
    p.add_argument("--cwd", default="/data/home/acw723/para-audio-id")
    p.add_argument("--control-path", default="~/.ssh/apocrita-%C")
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    ssh = ["ssh", "-o", "ControlMaster=auto", "-o", "ControlPersist=600",
           "-o", f"ControlPath={a.control_path}", "apocrita"]
    job = a.job
    if job is None:
        rows = subprocess.check_output(ssh + [
            '/opt/slurm/bin/squeue -h -u "$USER" -t RUNNING -o "%i|%j"'], text=True)
        jobs = [int(row.split("|", 1)[0]) for row in rows.splitlines()
                if "|ood_jupy" in row and row.split("|", 1)[0].isdigit()]
        if not jobs:
            p.error("No running ood_jupy allocation found")
        job = max(jobs)
    command = a.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        p.error("Supply a command after --")
    # All computation enters a scheduler step; direct compute-node SSH is avoided.
    body = ("export PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            "OPENBLAS_NUM_THREADS=1; cd " + shlex.quote(a.cwd) + " && "
            + shlex.join(command))
    remote = shlex.join(["/opt/slurm/bin/srun", f"--jobid={job}", "--overlap",
                         "--nodes=1", "--ntasks=1", "/bin/bash", "-lc", body])
    print(f"Using Apocrita allocation {job}", file=sys.stderr, flush=True)
    return subprocess.call(ssh + [remote])


if __name__ == "__main__":
    raise SystemExit(main())
