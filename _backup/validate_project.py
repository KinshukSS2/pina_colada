#!/usr/bin/env python3
"""Static and optional runtime validator for the Traffic Control OpenEnv project.

Default behavior:
- checks file presence
- checks core content markers
- checks Python syntax
- checks requirements and docs

Optional behavior:
- --run-openenv-validate   run `openenv validate`
- --run-docker-build       run `docker build -t traffic-env .`
- --run-docker-smoke       run `docker run -d -p 7860:7860 traffic-env` and probe localhost:7860/

Exit code:
- 0 if all required checks pass
- non-zero if any required check fails
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT
if (ROOT / 'traffic_openenv_checklist.md').exists() and (ROOT / 'validate_project.py').exists():
    # If the script is copied into the repo root, this still works.
    REPO_ROOT = ROOT

REQUIRED_FILES = [
    'README.md',
    'Dockerfile',
    'requirements.txt',
    'openenv.yaml',
    'inference.py',
    'env/models.py',
    'env/environment.py',
    'env/simulator.py',
    'env/reward.py',
    'env/grader.py',
    'env/tasks.py',
    'server/main.py',
]

REQUIRED_REQS = ['fastapi', 'uvicorn', 'pydantic', 'openenv-core', 'huggingface_hub']


def rel(path: str) -> Path:
    return REPO_ROOT / path


def ok(msg: str) -> None:
    print(f'[PASS] {msg}')


def warn(msg: str) -> None:
    print(f'[WARN] {msg}')


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
    markers = [
        ('traffic', 'traffic domain'),
        ('action space', 'action space'),
        ('observation space', 'observation space'),
        ('reward', 'reward'),
        ('task', 'tasks'),
        ('setup', 'setup'),
        ('baseline', 'baseline'),
        ('docker', 'docker'),
        ('hugging face', 'hugging face'),
        ('openenv', 'openenv'),
    ]
    good = True
    for req in REQUIRED_REQS:
        if req.lower() not in txt:
            fail(f'requirements.txt missing dependency marker: {req}')
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
            fail(f'README.md missing topic marker: {m}')
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
            fail(f'openenv.yaml missing field marker: {m}')
            good = False
        else:
            ok(f'openenv.yaml includes {m}')
    # task names
    for t in ['easy', 'medium', 'hard']:
        if t not in txt:
            fail(f'openenv.yaml missing task marker: {t}')
            good = False
        else:
            ok(f'openenv.yaml includes task marker: {t}')
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
            fail(f'Dockerfile missing: {label} ({needle})')
            good = False
        else:
            ok(f'Dockerfile includes {label}')
    return good


def check_inference() -> bool:
    p = rel('inference.py')
    if not p.exists():
        fail('inference.py missing at repo root')
        return False
    txt = read_text(p)
    lower = txt.lower()
    markers = [
        ('API_BASE_URL', 'reads API_BASE_URL'),
        ('MODEL_NAME', 'reads MODEL_NAME'),
        ('hold', 'has fallback action'),
    ]
    good = True
    for needle, label in markers:
        if needle not in txt and needle.lower() not in lower:
            fail(f'inference.py missing: {label} ({needle})')
            good = False
        else:
            ok(f'inference.py {label}')

    if 'OPENAI_API_KEY' in txt or 'GEMINI_API_KEY' in txt or 'HF_TOKEN' in txt:
        ok('inference.py reads an API key variable')
    else:
        fail('inference.py missing: reads API key variable (OPENAI_API_KEY/GEMINI_API_KEY/HF_TOKEN)')
        good = False

    if 'client.chat.completions.create' in lower or 'generatecontent' in lower:
        ok('inference.py has an LLM generation call path')
    else:
        fail('inference.py missing: LLM generation call path')
        good = False
    return good


def check_server() -> bool:
    p = rel('server/main.py')
    if not p.exists():
        fail('server/main.py missing')
        return False
    txt = read_text(p).lower()
    markers = [
        ('fastapi', 'FastAPI import'),
        ('/reset', '/reset endpoint'),
        ('/step', '/step endpoint'),
        ('/state', '/state endpoint'),
        ('0.0.0.0', 'bind host marker'),
        ('7860', 'port marker'),
    ]
    good = True
    for needle, label in markers:
        if needle not in txt:
            fail(f'server/main.py missing: {label}')
            good = False
        else:
            ok(f'server/main.py has {label}')
    return good


def check_env_modules() -> bool:
    good = True
    modules = {
        'env/models.py': ['BaseModel', 'Observation', 'Action'],
        'env/environment.py': ['reset', 'step', 'state'],
        'env/simulator.py': [],
        'env/reward.py': ['compute_reward'],
        'env/grader.py': ['compute_score'],
        'env/tasks.py': ['easy', 'medium', 'hard'],
    }
    for file, markers in modules.items():
        p = rel(file)
        if not p.exists():
            fail(f'Missing module: {file}')
            good = False
            continue
        txt = read_text(p)
        for m in markers:
            if m not in txt and m.lower() not in txt.lower():
                fail(f'{file} missing marker: {m}')
                good = False
            else:
                ok(f'{file} has marker: {m}')
    return good


def check_task_count() -> bool:
    p = rel('env/tasks.py')
    if not p.exists():
        return False
    txt = read_text(p).lower()
    count = 0
    for t in ['easy', 'medium', 'hard']:
        if t in txt:
            count += 1
    if count >= 3:
        ok('At least 3 task markers found (easy/medium/hard)')
        return True
    fail('Fewer than 3 task markers found in env/tasks.py')
    return False


def check_py_files_compilation() -> bool:
    py_files = [
        'inference.py',
        'env/models.py',
        'env/environment.py',
        'env/simulator.py',
        'env/reward.py',
        'env/grader.py',
        'env/tasks.py',
        'server/main.py',
    ]
    existing = [f for f in py_files if rel(f).exists()]
    return check_python_syntax(existing)


def try_run(cmd: List[str], cwd: Path = REPO_ROOT, timeout: int = 120) -> Tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    out = (proc.stdout or '') + (proc.stderr or '')
    return proc.returncode, out.strip()


def check_tool(name: str) -> bool:
    if shutil.which(name):
        ok(f'Tool available: {name}')
        return True
    warn(f'Tool not found: {name}')
    return False


def run_openenv_validate() -> bool:
    if not shutil.which('openenv'):
        warn('Skipping openenv validate: openenv CLI not found')
        return True
    code, out = try_run(['openenv', 'validate'], timeout=180)
    print(out)
    if code == 0:
        ok('openenv validate passed')
        return True
    fail('openenv validate failed')
    return False


def run_docker_build() -> bool:
    if not shutil.which('docker'):
        warn('Skipping docker build: docker not found')
        return True
    code, out = try_run(['docker', 'build', '-t', 'traffic-env', '.'], timeout=1800)
    print(out)
    if code == 0:
        ok('docker build passed')
        return True
    fail('docker build failed')
    return False


def run_docker_smoke() -> bool:
    if not shutil.which('docker'):
        warn('Skipping docker smoke test: docker not found')
        return True
    # Start container in detached mode
    try:
        code, out = try_run(['docker', 'run', '-d', '-p', '7860:7860', 'traffic-env'], timeout=30)
        if code != 0:
            fail('docker run failed')
            print(out)
            return False
        container_id = out.splitlines()[-1].strip()
        ok(f'docker container started: {container_id[:12]}')
        # Simple HTTP probe
        import urllib.request
        import time
        url = 'http://127.0.0.1:7860/'
        for _ in range(20):
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if 200 <= resp.status < 300:
                        ok('Docker smoke HTTP check passed')
                        break
            except Exception:
                time.sleep(2)
        else:
            fail('Docker smoke HTTP check failed')
            return False
        return True
    finally:
        # Best-effort cleanup
        try:
            subprocess.run(['docker', 'rm', '-f', 'traffic-env'], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30)
        except Exception:
            pass


def check_env_vars() -> bool:
    base_vars = ['API_BASE_URL', 'MODEL_NAME']
    good = True
    for var in base_vars:
        if os.getenv(var):
            ok(f'Environment variable set: {var}')
        else:
            warn(f'Environment variable not set: {var} (defaults exist)')

    if os.getenv('OPENAI_API_KEY') or os.getenv('GEMINI_API_KEY') or os.getenv('HF_TOKEN'):
        ok('Environment variable set: OPENAI_API_KEY or GEMINI_API_KEY or HF_TOKEN')
    else:
        warn('No API key found in OPENAI_API_KEY/GEMINI_API_KEY/HF_TOKEN (required only for running inference.py)')

    return good


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate the Traffic Control OpenEnv project.')
    parser.add_argument('--run-openenv-validate', action='store_true', help='Run `openenv validate` if available')
    parser.add_argument('--run-docker-build', action='store_true', help='Run `docker build -t traffic-env .` if Docker is available')
    parser.add_argument('--run-docker-smoke', action='store_true', help='Run a Docker smoke test after build')
    parser.add_argument('--check-tools', action='store_true', help='Check whether docker/openenv are installed')
    args = parser.parse_args()

    print('Traffic Control OpenEnv Project Validator')
    print(f'Workspace: {REPO_ROOT}')
    print()

    required_ok = True
    for path in REQUIRED_FILES:
        required_ok &= check_exists(path)

    required_ok &= check_requirements()
    required_ok &= check_readme()
    required_ok &= check_openenv_yaml()
    required_ok &= check_dockerfile()
    required_ok &= check_inference()
    required_ok &= check_server()
    required_ok &= check_env_modules()
    required_ok &= check_task_count()
    required_ok &= check_py_files_compilation()
    required_ok &= check_env_vars()

    if args.check_tools:
        check_tool('docker')
        check_tool('openenv')

    if args.run_openenv_validate:
        required_ok &= run_openenv_validate()

    if args.run_docker_build:
        required_ok &= run_docker_build()

    if args.run_docker_smoke:
        required_ok &= run_docker_smoke()

    print()
    if required_ok:
        ok('All required checks passed')
        return 0
    fail('One or more required checks failed')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
