#!/usr/bin/env python3
"""Minimal HAD-MC Studio controller for an isolated DCU deployment."""

from __future__ import annotations

import json
import os
import ssl
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT = Path(os.environ.get("HADMC_ROOT", "/data/hadmc"))
STATIC = Path(__file__).parent / "static"
JOBS = ROOT / "artifacts" / "jobs"
K8S_HOST = os.environ.get("KUBERNETES_SERVICE_HOST")
K8S_PORT = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
K8S_NAMESPACE = os.environ.get("HADMC_NAMESPACE", "hadmc")
K8S_TOKEN = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
K8S_CA = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
JOB_IMAGE = os.environ.get("HADMC_JOB_IMAGE", "10.8.144.65/platform/llama-factory-trainer-dcu:v1.17-isolated-compat")
REAL_QWEN_POLICIES = (
    {"id": "quality-guard", "ratio": 0.10, "weights": (0.70, 0.20, 0.10)},
    {"id": "balanced", "ratio": 0.25, "weights": (0.55, 0.30, 0.15)},
    {"id": "storage-first", "ratio": 0.40, "weights": (0.35, 0.50, 0.15)},
    {"id": "latency-seeker", "ratio": 0.55, "weights": (0.35, 0.30, 0.35)},
)


def kubernetes_enabled() -> bool:
    return bool(K8S_HOST and K8S_TOKEN.exists() and K8S_CA.exists())


def kube_json(path: str, method: str = "GET", payload: dict | None = None) -> dict:
    if not kubernetes_enabled():
        raise RuntimeError("Kubernetes service account is unavailable")
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        f"https://{K8S_HOST}:{K8S_PORT}{path}",
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {K8S_TOKEN.read_text(encoding='utf-8').strip()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    context = ssl.create_default_context(cafile=str(K8S_CA))
    with urlopen(request, context=context, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def job_records(limit: int | None = 30) -> list[dict]:
    if kubernetes_enabled():
        payload = kube_json(f"/apis/batch/v1/namespaces/{K8S_NAMESPACE}/jobs")
        records: list[dict] = []
        for item in payload.get("items", []):
            metadata = item.get("metadata", {})
            labels = metadata.get("labels", {})
            name = metadata.get("name", "")
            is_smoke = name.startswith("hadmc-dcu-runtime-smoke-")
            if labels.get("app.kubernetes.io/name") != "had-mc" and not is_smoke:
                continue
            status_data = item.get("status", {})
            status = "running"
            if status_data.get("failed"):
                status = "failed"
            elif status_data.get("succeeded"):
                status = "completed"
            node = labels.get("hadmc.io/node")
            if not node:
                node = "gpu-01" if "gpu01" in name else "gpu-02" if "gpu02" in name else "unknown"
            records.append({
                "id": name,
                "kind": labels.get("hadmc.io/kind", "runtime-smoke" if is_smoke else "unknown"),
                "node": node,
                "policy": labels.get("hadmc.io/policy") or None,
                "status": status,
                "started_at": status_data.get("startTime") or metadata.get("creationTimestamp"),
                "finished_at": status_data.get("completionTime"),
            })
        records = sorted(records, key=lambda item: item.get("started_at") or "", reverse=True)
        return records if limit is None else records[:limit]
    return []


def kubernetes_status() -> dict:
    jobs = job_records(limit=None)
    nodes: list[dict] = []
    for node in ("gpu-01", "gpu-02"):
        smoke = next((job for job in jobs if job["kind"] == "runtime-smoke" and job["node"] == node), None)
        ready = bool(smoke and smoke["status"] == "completed")
        nodes.append({
            "hostname": node,
            "runtime": "DTK/HYHAL container runtime",
            "dcu": {"ok": ready, "allocatable_units": 16, "product": "Z200SM_71"},
            "torch": {"available": ready, "version": "2.4.1 / HIP 6.3" if ready else None, "devices": 1 if ready else 0},
        })
    return {
        "nodes": nodes,
        "accelerator_inventory": [
            {"kind": "DCU", "status": "connected", "nodes": ["gpu-01", "gpu-02"], "detail": "2 nodes; Kubernetes reports 16 allocatable DCU units per node"},
            {"kind": "GPU", "status": "not_connected", "nodes": [], "detail": "adapter contract reserved; no NVIDIA GPU resource is registered"},
            {"kind": "NPU", "status": "not_connected", "nodes": [], "detail": "adapter contract reserved; no NPU resource is registered"},
        ],
    }


def job_manifest(kind: str, node: str, policy: dict | None = None) -> dict:
    suffix = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    policy_id = policy["id"] if policy else None
    name = f"hadmc-{kind}-{policy_id + '-' if policy_id else ''}{node}-{suffix}"
    if kind == "probe":
        command = "python3 /workspace/r3_revision/code/platform_probe.py --platform-tag dcu --output-dir /artifacts/jobs/$HADMC_JOB_ID"
    elif kind == "qwen-demo":
        command = "python3 -u /workspace/studio/qwen_compression_demo.py --platform-tag dcu --output-dir /artifacts/jobs/$HADMC_JOB_ID"
    elif kind == "qwen-real":
        quality, storage, latency = policy["weights"]
        command = (
            "python3 -u /workspace/studio/qwen_real_compression.py "
            "--output-dir /artifacts/jobs/$HADMC_JOB_ID "
            f"--candidate-id {policy['id']} --prune-ratio {policy['ratio']} "
            f"--quality-weight {quality} --storage-weight {storage} --latency-weight {latency}"
        )
    else:
        command = "python3 -u /workspace/r3_revision/code/hadmc_experiments_complete.py --platform-tag dcu --results-dir /artifacts/jobs/$HADMC_JOB_ID --allow-missing-financial"
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name, "namespace": K8S_NAMESPACE, "labels": {
            "app.kubernetes.io/name": "had-mc", "app.kubernetes.io/part-of": "had-mc",
            "hadmc.io/kind": kind, "hadmc.io/node": node, "hadmc.io/policy": policy_id or "",
        }},
        "spec": {"backoffLimit": 0, "ttlSecondsAfterFinished": 604800, "template": {"metadata": {"labels": {
            "app.kubernetes.io/name": "had-mc", "hadmc.io/kind": kind, "hadmc.io/node": node, "hadmc.io/policy": policy_id or "",
        }}, "spec": {"restartPolicy": "Never", "nodeSelector": {"kubernetes.io/hostname": node}, "containers": [{
            "name": "hadmc", "image": JOB_IMAGE, "imagePullPolicy": "IfNotPresent",
            "resources": {"requests": {"hygon.com/dcunum": "1"}, "limits": {"hygon.com/dcunum": "1"}},
            "env": [
                {"name": "HADMC_JOB_ID", "value": name},
                {"name": "LD_LIBRARY_PATH", "value": "/opt/dtk/lib:/opt/dtk/hipblas/lib:/opt/hyhal/lib:/usr/lib64:/lib64:/usr/lib:/lib"},
                {"name": "ROCR_VISIBLE_DEVICES", "value": "0"},
                {"name": "HSA_OVERRIDE_GFX_VERSION", "value": "9.0.6"},
                {"name": "HF_ENDPOINT", "value": "https://hf-mirror.com"},
                {"name": "HF_HUB_DISABLE_XET", "value": "1"},
            ],
            "command": ["/bin/bash", "-lc"], "args": [f"set -euo pipefail; mkdir -p /artifacts/jobs/$HADMC_JOB_ID; {command}"],
            "volumeMounts": [
                {"name": "source", "mountPath": "/workspace", "readOnly": True}, {"name": "artifacts", "mountPath": "/artifacts"},
                {"name": "models", "mountPath": "/models"},
                {"name": "dcu-kfd", "mountPath": "/dev/kfd"}, {"name": "dcu-dri", "mountPath": "/dev/dri"},
                {"name": "dcu-mkfd", "mountPath": "/dev/mkfd"}, {"name": "dcu-hyhal", "mountPath": "/opt/hyhal"},
                {"name": "dshm", "mountPath": "/dev/shm"},
            ],
        }], "volumes": [
            {"name": "source", "hostPath": {"path": "/data/hadmc/source/HAD-MC", "type": "Directory"}},
            {"name": "artifacts", "hostPath": {"path": "/data/hadmc/artifacts", "type": "Directory"}},
            {"name": "models", "hostPath": {"path": "/data/hadmc/models", "type": "DirectoryOrCreate"}},
            {"name": "dcu-kfd", "hostPath": {"path": "/dev/kfd", "type": "CharDevice"}},
            {"name": "dcu-dri", "hostPath": {"path": "/dev/dri", "type": "Directory"}},
            {"name": "dcu-mkfd", "hostPath": {"path": "/dev/mkfd", "type": "CharDevice"}},
            {"name": "dcu-hyhal", "hostPath": {"path": "/opt/hyhal", "type": "Directory"}},
            {"name": "dshm", "emptyDir": {"medium": "Memory"}},
        ]}}},
    }


def real_qwen_runs() -> list[dict]:
    records: list[dict] = []
    if not JOBS.exists():
        return records
    statuses = {record["id"]: record for record in job_records()}
    for directory in sorted(JOBS.glob("hadmc-qwen-real-*"), key=lambda item: item.stat().st_mtime, reverse=True):
        progress_file = directory / "QLIGHT_REAL_QWEN_PROGRESS.json"
        result_file = directory / "QLIGHT_REAL_QWEN_CANDIDATE.json"
        progress_data: dict = {}
        result_data: dict = {}
        try:
            if progress_file.exists():
                progress_data = json.loads(progress_file.read_text(encoding="utf-8"))
            if result_file.exists():
                result_data = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        status = statuses.get(directory.name)
        if status is None:
            continue
        candidate = result_data.get("candidate", progress_data.get("candidate", {}))
        policy = candidate.get("candidate_id") or status.get("policy")
        if not policy:
            policy = directory.name
        records.append({
            "job_id": directory.name,
            "policy": policy,
            "node": status.get("node", "unknown"),
            "job_status": status.get("status", "archived"),
            "phase": progress_data.get("phase", "queued"),
            "updated_at_utc": progress_data.get("updated_at_utc"),
            "error": progress_data.get("error"),
            "candidate": candidate,
            "model": result_data.get("model", {}),
            "evidence_scope": result_data.get("evidence_scope", {}),
            "artifact": str(directory.relative_to(ROOT / "artifacts")),
        })
    return records[:24]


class StudioHandler(SimpleHTTPRequestHandler):
    server_version = "HADMCStudio/0.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        return

    def send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            if not kubernetes_enabled():
                self.send_json({"ok": False, "error": "Kubernetes service account is unavailable"}, HTTPStatus.SERVICE_UNAVAILABLE)
                return
            try:
                self.send_json({"ok": True, "data": kubernetes_status()})
            except (OSError, RuntimeError, ValueError) as exc:
                self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_GATEWAY)
            return
        if path == "/api/jobs":
            self.send_json({"ok": True, "data": job_records()})
            return
        if path == "/api/artifacts":
            artifacts = [
                str(item.relative_to(ROOT / "artifacts"))
                for item in sorted((ROOT / "artifacts").rglob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:40]
            ] if (ROOT / "artifacts").exists() else []
            self.send_json({"ok": True, "data": artifacts})
            return
        if path == "/api/qwen-results":
            results: list[dict] = []
            artifacts_root = ROOT / "artifacts"
            if artifacts_root.exists():
                for item in artifacts_root.rglob("QLIGHT_QWEN_DEMO_RESULTS.json"):
                    try:
                        payload = json.loads(item.read_text(encoding="utf-8"))
                        payload["artifact"] = str(item.relative_to(artifacts_root))
                        results.append(payload)
                    except (OSError, ValueError):
                        continue
            results.sort(key=lambda item: item.get("completed_at_utc", ""), reverse=True)
            self.send_json({"ok": True, "data": results[:12]})
            return
        if path == "/api/real-qwen-runs":
            self.send_json({"ok": True, "data": real_qwen_runs()})
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/jobs":
            self.send_json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        size = int(self.headers.get("Content-Length", "0"))
        if size > 4096:
            self.send_json({"ok": False, "error": "request too large"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            request = json.loads(self.rfile.read(size).decode("utf-8"))
            kind = request["kind"]
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
            self.send_json({"ok": False, "error": "invalid request"}, HTTPStatus.BAD_REQUEST)
            return
        node = request.get("node", "gpu-02")
        if kind not in {"probe", "full", "qwen-demo", "qwen-real-campaign"} or node not in {"gpu-01", "gpu-02"}:
            self.send_json({"ok": False, "error": "unsupported job kind"}, HTTPStatus.BAD_REQUEST)
            return
        if kubernetes_enabled():
            try:
                if kind == "qwen-real-campaign":
                    created_jobs = []
                    for index, policy in enumerate(REAL_QWEN_POLICIES):
                        target_node = ("gpu-01", "gpu-02")[index % 2]
                        created = kube_json(
                            f"/apis/batch/v1/namespaces/{K8S_NAMESPACE}/jobs",
                            "POST",
                            job_manifest("qwen-real", target_node, policy),
                        )
                        created_jobs.append({"id": created.get("metadata", {}).get("name"), "node": target_node, "policy": policy["id"]})
                    self.send_json({"ok": True, "data": {"kind": kind, "jobs": created_jobs, "status": "accepted"}}, HTTPStatus.ACCEPTED)
                    return
                created = kube_json(f"/apis/batch/v1/namespaces/{K8S_NAMESPACE}/jobs", "POST", job_manifest(kind, node))
                metadata = created.get("metadata", {})
                self.send_json({"ok": True, "data": {"id": metadata.get("name"), "kind": kind, "node": node, "status": "accepted"}}, HTTPStatus.ACCEPTED)
            except (OSError, RuntimeError, ValueError) as exc:
                self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_GATEWAY)
            return
        self.send_json({"ok": False, "error": "Kubernetes service account is unavailable"}, HTTPStatus.SERVICE_UNAVAILABLE)


if __name__ == "__main__":
    host = os.environ.get("HADMC_LISTEN_HOST", "0.0.0.0")
    port = int(os.environ.get("HADMC_LISTEN_PORT", "18888"))
    ThreadingHTTPServer((host, port), StudioHandler).serve_forever()
