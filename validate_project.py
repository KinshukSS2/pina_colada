#!/usr/bin/env python3
"""Static and optional runtime validator for the Ticket Triage OpenEnv project."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    'README.md',
    'Dockerfile',
    'requirements.txt',
    'openenv.yaml',
    'inference.py',
    'env/schemas.py',
    'env/environment.py',
    'env/simulator.py',
    'env/reward.py',
    'env/__init__.py',
    'config/env_config.py',
    'config/task_configs.py',
    'graders/base_grader.py',
    'graders/easy_grader.py',
    'graders/medium_grader.py',
    'graders/hard_grader.py',
    'tasks/task_easy.py',
    'tasks/task_medium.py',
    'tasks/task_hard.py',
    'server/app.py',
    'server/main.py',
]

REQUIRED_REQS = ['fastapi', 'uvicorn', 'pydantic', 'openenv-core', 'huggingface_hub']


def rel(path: str) -> Path:
    return ROOT / path


def ok(msg: str) -> None:
    print(f'[PASS] {msg}')


def fail(msg: str) -> None:
    print(f'[FAIL] {msg}')


def read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='ignore')


def check_exists(path: str) -> bool:
    p = rel(path)
    if p.exists():
        ok(f'Exists: {path}')
        return True
    fail(f'Missing required file: {path}')
    return False


def check_python_syntax(files: Iterable[str]) -> bool:
    good = True
    for f in files:
        p = rel(f)
        if not p.exists():
            good = False
            continue
        try:
            compile(read_text(p), str(p), 'exec')
            ok(f'Syntax OK: {f}')
        except SyntaxError as exc:
            fail(f'Syntax error in {f}: {exc}')
            good = False
    return good


def check_requirements() -> bool:
    p = rel('requirements.txt')
    if not p.exists():
        fail('requirements.txt missing')
        return False
    txt = read_text(p).lower()
    good = True
    for req in REQUIRED_REQS:
        if req.lower() not in txt:
            fail(f'requirements.txt missing dependency: {req}')
            good = False
        else:
            ok(f'requirements.txt includes {req}')
    return good


def check_readme() -> bool:
    p = rel('README.md')
    if not p.exists():
        fail('README.md missing')
        return False
    txt = read_text(p).lower()
    markers = [
        'environment description', 'action space', 'observation space', 'reward',
        'tasks', 'setup', 'baseline', 'docker', 'hugging face', 'openenv'
    ]
    good = True
    for m in markers:
        if m not in txt:
            fail(f'README.md missing topic: {m}')
            good = False
        else:
            ok(f'README covers: {m}')
    return good


def check_openenv_yaml() -> bool:
    p = rel('openenv.yaml')
    if not p.exists():
        fail('openenv.yaml missing')
        return False
    txt = read_text(p).lower()
    markers = ['name:', 'description:', 'entry_point:', 'actions:', 'observations:', 'tasks:']
    good = True
    for m in markers:
        if m not in txt:
            fail(f'openenv.yaml missing: {m}')
            good = False
        else:
            ok(f'openenv.yaml includes {m}')
    for t in ['easy', 'medium', 'hard']:
        if t not in txt:
            fail(f'openenv.yaml missing task: {t}')
            good = False
        else:
            ok(f'openenv.yaml includes task: {t}')
    return good


def check_dockerfile() -> bool:
    p = rel('Dockerfile')
    if not p.exists():
        fail('Dockerfile missing')
        return False
    txt = read_text(p).lower()
    checks = [
        ('python', 'base image'),
        ('copy', 'copy source'),
        ('pip install', 'install deps'),
        ('expose 7860', 'exposes port 7860'),
        ('uvicorn', 'starts uvicorn'),
        ('0.0.0.0', 'binds to 0.0.0.0'),
    ]
    good = True
    for needle, label in checks:
        if needle not in txt:
            fail(f'Dockerfile missing: {label}')
            good = False
        else:
            ok(f'Dockerfile includes {label}')
    return good


def check_inference() -> bool:
    p = rel('inference.py')
    if not p.exists():
        fail('inference.py missing')
        return False
    txt = read_text(p).lower()
    markers = ['openai', 'legacy/reset', 'legacy/step', 'run_episode']
    good = True
    for m in markers:
        if m not in txt:
            fail(f'inference.py missing: {m}')
            good = False
        else:
            ok(f'inference.py includes: {m}')
    return good


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-openenv-validate', action='store_true')
    parser.add_argument('--run-docker-build', action='store_true')
    args = parser.parse_args()

    all_ok = True

    py_files = [f for f in REQUIRED_FILES if f.endswith('.py')]

    for f in REQUIRED_FILES:
        if not check_exists(f):
            all_ok = False

    if not check_python_syntax(py_files):
        all_ok = False

    if not check_requirements():
        all_ok = False

    if not check_readme():
        all_ok = False

    if not check_openenv_yaml():
        all_ok = False

    if not check_dockerfile():
        all_ok = False

    if not check_inference():
        all_ok = False

    if args.run_openenv_validate:
        try:
            result = subprocess.run(['openenv', 'validate'], capture_output=True, text=True, cwd=ROOT, check=False)
            if result.returncode == 0:
                ok('openenv validate passed')
            else:
                fail(f'openenv validate failed: {result.stderr}')
                all_ok = False
        except FileNotFoundError:
            fail('openenv CLI not found')

    if args.run_docker_build:
        try:
            result = subprocess.run(['docker', 'build', '-t', 'ticket-triage', '.'], capture_output=True, text=True, cwd=ROOT, check=False)
            if result.returncode == 0:
                ok('Docker build succeeded')
            else:
                fail(f'Docker build failed: {result.stderr}')
                all_ok = False
        except FileNotFoundError:
            fail('docker CLI not found')

    if all_ok:
        print('\n[RESULT] All checks PASSED')
        return 0
    else:
        print('\n[RESULT] Some checks FAILED')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
