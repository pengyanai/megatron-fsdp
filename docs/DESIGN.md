# 设计:mcore 剥离面

本仓是 `NVIDIA/Megatron-LM` 里 `megatron/core/distributed/fsdp/src/` 的原样副本,
只改了打包相关的呈现(README / pyproject 引用),**没有改动任何 Python 源码**。
这里记录"为什么这样切"和"切在哪儿"。

## 1. 剥离边界

上游目录:

```
megatron/core/distributed/fsdp/
├── __init__.py              ← 只导出 mcore_fsdp_adapter.FullyShardedDataParallel
├── mcore_fsdp_adapter.py    ← 依赖 mcore,本仓不带
└── src/                     ← 独立包本体,本仓全部带过来
    ├── pyproject.toml       ← 上游自己就准备了独立打包配置(name = "megatron-fsdp")
    ├── README.md            → 本仓移到 docs/UPSTREAM_API.md
    ├── docs/
    └── megatron_fsdp/
```

**切分判据**:`src/` 是上游为独立发布准备的自包含单元(`pyproject.toml` 里
`include = ["megatron_fsdp", "megatron_fsdp.*"]`,PyPI 上的 `megatron-fsdp` 就是它)。
`mcore_fsdp_adapter.py` 反过来是"让 mcore 用 megatron_fsdp"的胶水,依赖方向相反,所以不带。

## 2. 为什么 megatron_fsdp 本身能脱离 mcore

包内对 `megatron.core` 的引用共 6 处,**每一处都在 `try/except ImportError` 内**:

| 位置 | 引用的 mcore 符号 | 回退 |
|---|---|---|
| `megatron_fsdp.py:56` | `DistributedDataParallelConfig` | 本包 `distributed_data_parallel_config.py`(同内容的 242 行副本) |
| `megatron_fsdp.py:59` | `is_submodule` | 本包 `utils.py:96` |
| `param_and_grad_buffer.py:69` | `DistributedDataParallelConfig` | 同上副本 |
| `param_and_grad_buffer.py:72` | `get_cuda_rng_tracker` | 本包 `utils.py:490` |
| `param_and_grad_buffer.py:98` | `megatron.core.nccl_allocator` | `apex.contrib.nccl_allocator` → `None` |
| `fully_shard.py:33` | `DistributedDataParallelConfig` | 同上副本 |

`utils.py:153` 还有一处 `find_megatron_fsdp()` 里的自引用,失败时返回 `None`,是纯探测辅助。

上游自己的注释也说明了这是有意为之:

```python
# fully_shard.py:68
# TODO(@cspades): Copied from megatron.core.utils to avoid depending on MCore
# for Megatron-FSDP. Should remove when the API is no longer experimental.
```

运行时日志会打印 `"Megatron Core is not installed, Megatron-FSDP will run without Megatron Core."`。

唯一实质降级是 NCCL allocator:没有 mcore 也没有 apex 时 `NCCL_ALLOCATOR = None`,
失去 NCCL user-buffer 注册(性能优化),功能不受影响。

## 3. 依赖分层

- **必需**:`torch`(DeviceMesh / DTensor / Distributed Checkpoint)、`einops`、`packaging`
- **可选**:`transformer_engine`(FP8 / MXFP8 / NVFP4 路径,`HAVE_TE` 守卫)、
  `apex`(NCCL allocator 第二级回退、multi-tensor apply)、`emerging-optimizers`(Muon)

## 4. 已知边界

- `MegatronFSDP.__init__`(`megatron_fsdp.py:240`)无条件调用 `torch.cuda.current_device()`,
  所以**显式传 `device="cpu"` 也会在纯 CPU 主机上崩**。CPU 上只能 import,不能跑训练步。
- 自动建 mesh 路径写死 `device_type="cuda"`(`fully_shard.py:318`)。
- 首次调用会加载 `torch.distributed.elastic`(经 `utils.py` 的依赖链),macOS 上会打印
  "Redirects are currently not supported in MacOs." 的无害告警。

## 5. 与本仓相关的取舍

- 单测没有搬过来:上游 `tests/unit_tests/distributed/mfsdp_v1|v2` 里多数文件 `import megatron.core`,
  搬过来要逐个改写才有意义,属于后续可选工作。
- 本仓不做任何性能改动。要改实现请在 `megatron_fsdp/` 上直接改,并在 commit message 里注明
  相对上游 commit 的偏离,方便将来跟进上游。

## 6. 更新上游

```bash
git -C ~/OpenSource/Megatron-LM fetch
# 对比旧 commit 到新 commit 之间 src/ 的差异
git -C ~/OpenSource/Megatron-LM log --oneline OLD..NEW -- megatron/core/distributed/fsdp/src
rsync -a --delete ~/OpenSource/Megatron-LM/megatron/core/distributed/fsdp/src/ \
      ~/PersonalCode/pengyanai/megatron-fsdp/ \
      --exclude README.md --exclude pyproject.toml --exclude .git
```

同步后更新 README 顶部的 commit 与版本号,并重跑 `tests/smoke_no_mcore.py`。
