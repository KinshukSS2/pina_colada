#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
SERVER_PORT="${SERVER_PORT:-7861}"
SERVER_URL="http://127.0.0.1:${SERVER_PORT}"
OPENROUTER_URL="https://openrouter.ai/api/v1"
MODEL_NAME="${MODEL_NAME:-openai/gpt-4o-mini}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[FAIL] Missing virtualenv python at ${PYTHON_BIN}"
  echo "Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[FAIL] OPENAI_API_KEY is not set in this shell"
  echo "Run: export OPENAI_API_KEY='sk-or-v1-...your real key...'"
  exit 1
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m uvicorn server.main:app --host 0.0.0.0 --port "${SERVER_PORT}" >/tmp/meta_server.log 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 40); do
  if curl -fsS "${SERVER_URL}/" >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done

if ! curl -fsS "${SERVER_URL}/" >/dev/null 2>&1; then
  echo "[FAIL] Server failed to start on ${SERVER_URL}"
  tail -n 80 /tmp/meta_server.log || true
  exit 1
fi

declare -a TASKS=(easy medium hard)

for t in "${TASKS[@]}"; do
  log_file="/tmp/meta_${t}.log"
  echo "[RUN] task=${t}"
  ENV_BASE_URL="${SERVER_URL}" \
  API_BASE_URL="${OPENROUTER_URL}" \
  MODEL_NAME="${MODEL_NAME}" \
  OPENAI_API_KEY="${OPENAI_API_KEY}" \
  "${PYTHON_BIN}" inference.py "${t}" | tee "${log_file}"
done

echo
echo "| Task | Score | Steps | Session | Auth |"
echo "|---|---:|---:|---|---|"
for t in "${TASKS[@]}"; do
  log_file="/tmp/meta_${t}.log"
  end_line="$(grep '^\[END\]' "${log_file}" | tail -n1 || true)"
  score="$(echo "${end_line}" | sed -n 's/.*score=\([0-9.]*\).*/\1/p')"
  steps="$(echo "${end_line}" | sed -n 's/.*steps=\([0-9]*\).*/\1/p')"
  session="$(echo "${end_line}" | sed -n 's/.*session=\([^ ]*\).*/\1/p')"
  if grep -q "Missing Authentication header\|Error code: 401" "${log_file}"; then
    auth="401"
  else
    auth="ok"
  fi
  echo "| ${t} | ${score:-N/A} | ${steps:-N/A} | ${session:-N/A} | ${auth} |"
done

echo
echo "[DONE] Logs saved to /tmp/meta_easy.log, /tmp/meta_medium.log, /tmp/meta_hard.log"
