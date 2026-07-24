#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAMESPACE="${HADMC_NAMESPACE:-hadmc}"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is required; run this from the configured Kubernetes administration node." >&2
  exit 1
fi

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "$NAMESPACE" create configmap hadmc-studio-code \
  --from-file=server.py="$ROOT/studio/server.py" \
  --from-file=index.html="$ROOT/studio/static/index.html" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f "$ROOT/deploy/dcu/kubernetes/hadmc-studio.yaml"
kubectl -n "$NAMESPACE" rollout restart deployment/hadmc-studio
kubectl -n "$NAMESPACE" rollout status deployment/hadmc-studio --timeout=180s
printf 'HAD-MC Studio is ready on the private DCU endpoint: http://10.8.159.54:18888/\n'
