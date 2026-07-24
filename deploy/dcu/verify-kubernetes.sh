#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${HADMC_NAMESPACE:-hadmc}"
STUDIO_URL="${HADMC_STUDIO_URL:-http://10.8.159.54:18888}"

command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required" >&2; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 1; }

kubectl -n "$NAMESPACE" rollout status deployment/hadmc-studio --timeout=180s
kubectl -n "$NAMESPACE" get jobs

status="$(curl --fail --silent --show-error "$STUDIO_URL/api/status")"
jobs="$(curl --fail --silent --show-error "$STUDIO_URL/api/jobs")"

python3 - "$status" "$jobs" <<'PY'
import json
import sys

status = json.loads(sys.argv[1])
jobs = json.loads(sys.argv[2])
if not status.get("ok"):
    raise SystemExit("status API returned failure")
nodes = status.get("data", {}).get("nodes", [])
ready = [node for node in nodes if node.get("dcu", {}).get("ok") and node.get("torch", {}).get("available")]
if len(ready) != 2:
    raise SystemExit(f"expected two ready DCU nodes, received {len(ready)}")
if not jobs.get("ok"):
    raise SystemExit("jobs API returned failure")
print("verified: two DCU nodes are ready and Studio APIs are responding")
PY
