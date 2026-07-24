# QLight DCU deployment

This directory is the reproducible deployment layer for the Hygon DCU bare-metal environment. HAD-MC is isolated in its own `hadmc` Kubernetes namespace and does not alter existing Kubeflow, GPUStack, monitoring, or control-plane workloads.

## Topology

| Role | Host | Responsibility |
| --- | --- | --- |
| Kubernetes administration | `gpu-01` | Applies the namespace-scoped deployment and verifies cluster state. |
| Studio controller | `gpu-02` | Hosts the private Studio UI on port `18888` and creates only HAD-MC jobs. |
| Experiment workers | `gpu-01`, `gpu-02` | Each submitted job requests exactly one `hygon.com/dcunum` DCU allocation. |
| Shared state | `/data/hadmc` | Shared source, model cache, non-secret manifests, and experiment artifacts. |

The QLight Studio endpoint is intentionally private to the Supercomputing Internet VPN:

```text
http://10.8.159.54:18888/
```

Do not expose that endpoint to the public Internet without adding an authenticated reverse proxy and TLS. The controller is namespace-scoped and cannot read or change unrelated Kubernetes resources.

## Bootstrap

Run on the Kubernetes administration node after the source is present at `/data/hadmc/source/HAD-MC`:

```bash
cd /data/hadmc/source/HAD-MC
sudo bash deploy/dcu/bootstrap-kubernetes.sh
sudo bash deploy/dcu/verify-kubernetes.sh
```

`bootstrap-kubernetes.sh` creates the QLight Studio ConfigMap from this repository, applies the `hadmc` namespace/RBAC/Deployment manifest, and waits for the controller rollout. Re-running it is safe and updates only HAD-MC resources.

## Runtime contract

- Jobs use the proven DTK/HYHAL-compatible training image `llama-factory-trainer-dcu:v1.17-isolated-compat`.
- Each job mounts the original HAD-MC source read-only and writes artifacts to `/data/hadmc/artifacts/jobs/<job-id>`.
- The Studio `DCU probe` runs a PyTorch DCU matrix multiplication. The `R3 full suite` runs `r3_revision/code/hadmc_experiments_complete.py` with `--allow-missing-financial`; missing proprietary financial data is reported as skipped rather than fabricated.
- The Chinese QLight Studio shows real accelerator inventory. The current deployment has two connected Hygon DCU nodes (`gpu-01`, `gpu-02`); GPU and NPU adapter contracts are visible but are explicitly shown as not connected until matching Kubernetes resources are registered.
- `启动真实 Qwen 多策略压缩` uses the official `Qwen/Qwen2.5-7B-Instruct` checkpoint, cached at `/data/hadmc/models/qwen2.5-7b-instruct`. The task evaluates four HAD-MC multi-objective policy profiles across the two DCU nodes. Each job loads the real checkpoint, structurally prunes Qwen SwiGLU MLP channels, measures DCU forward latency, and writes `QLIGHT_REAL_QWEN_PROGRESS.json` plus `QLIGHT_REAL_QWEN_CANDIDATE.json`.
- Qwen checkpoint traffic uses `https://hf-mirror.com` because the DCU pod network does not have a Hugging Face IPv6 route. This is set only in the experiment-job environment; it does not create a general outbound proxy or expose the cluster.
- The real-Qwen results report exact FP16 parameter storage and a calibration logit-cosine agreement. They are not benchmark accuracy claims, do not imply fine-tuning, and do not claim INT8 kernels or acceleration unless those stages are separately executed and evidenced.
- The generic root `deploy.sh` is not valid for DCU deployment because its generic Python path does not validate the installed Hygon runtime.

## Acceptance

The baseline acceptance gate is:

1. `verify-kubernetes.sh` reports two ready DCU nodes.
2. A DCU probe completes on both `gpu-01` and `gpu-02` and writes a `platform_probe_dcu.json` artifact.
3. The QLight UI can submit and display a job through its browser controls.
4. A full R3 suite reaches a terminal `Complete` state with its `COMPLETE_EXPERIMENT_RESULTS.json` artifact preserved.
5. A real Qwen campaign reaches `Complete` on all four policy jobs and writes `QLIGHT_REAL_QWEN_CANDIDATE.json` for each policy; the QLight UI renders all measured candidates and selects the highest recorded reward without replacing the individual results.
