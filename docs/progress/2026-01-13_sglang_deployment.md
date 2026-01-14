# Progress: SGLang Deployment

**Date**: 2026-01-13
**Status**: Completed
**Phase**: Phase 5 - SGLang Deployment

---

## Summary

Successfully deployed SGLang inference server environment on A100 GPU VM, enabling high-performance batch inference for model evaluation.

---

## Environment Configuration

| Component | Value |
|-----------|-------|
| Environment Path | `/data/envs/sglang/` |
| Python Version | 3.11.14 |
| SGLang Version | 0.5.7 |
| Environment Size | ~11GB |
| Storage Location | Persistent data disk (`/data/`) |

---

## Storage Architecture Discovery

During deployment, discovered A100 VM's 4-layer storage architecture:

| Layer | Path | Capacity | Persistent | Use Case |
|-------|------|----------|------------|----------|
| 1 | `/data/` | 1TB | ✅ | Models, Checkpoints, Conda envs |
| 2 | `/mnt/shared/` | 50GB | ✅ | Code repos, configs |
| 3 | `/mnt/nvme/` | 880GB | ❌ | Temporary cache |
| 4 | `/` | 29GB | ✅ | System (avoid large installs) |

---

## Deployment Steps

1. **System Disk Cleanup**: Cleared space on root partition
2. **Conda Environment Creation**: Created Python 3.11 environment in `/data/envs/`
3. **SGLang Installation**: Installed via pip with all dependencies
4. **Verification**: Confirmed successful import and version

---

## Activation Commands

```bash
# Method 1: Direct activation (recommended)
source /data/envs/sglang/bin/activate

# Method 2: Conda activation
conda activate /data/envs/sglang

# Verification
python -c "import sglang; print(sglang.__version__)"
```

---

## Server Launch Commands

```bash
# Single GPU mode
python -m sglang.launch_server \
  --model-path /data/qwen3_or_debug_merged \
  --port 30000

# Dual GPU tensor parallel (A100×2)
python -m sglang.launch_server \
  --model-path /data/qwen3_or_debug_merged \
  --tensor-parallel-size 2 \
  --port 30000
```

---

## Performance Comparison

| Metric | Transformers | SGLang |
|--------|--------------|--------|
| Throughput | ~1 req/11s | ~10+ req/s |
| Batching | Sequential | Continuous |
| Memory | Single request | KV Cache sharing |
| API | Custom | OpenAI-compatible |

---

## Documentation Created

| File | Description |
|------|-------------|
| `~/.claude/docs/gpu_vm/storage_architecture.md` | 4-layer storage guide |
| `~/.claude/docs/gpu_vm/sglang_deployment.md` | SGLang deployment config |

---

## Next Steps

- [x] Document storage architecture
- [x] Add SGLang section to CLAUDE.md
- [ ] Use SGLang for large-scale evaluation (500+ samples)

---

**文档版本**: v1.0
**最后更新**: 2026-01-13
