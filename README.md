# megatron-fsdp

NVIDIA [Megatron-FSDP](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/distributed/fsdp/src)
的**独立 vendored 副本**:一个可以直接脱离 Megatron-Core 使用的 FSDP 实现,镜像一个固定上游 commit。

- 上游:`NVIDIA/Megatron-LM` @ `dc36b1a89f5de29b09875a03997fe6f1334e2a73`(包目录最后改动 `47715bb0`,2026-10-27)
- 包版本:`megatron_fsdp` 0.8.0rc0
- 许可:见 [LICENSE](LICENSE)(BSD-3 默认,内嵌 Apache-2.0;包内文件头为 Apache-2.0)

## 为什么有这个仓

`megatron_fsdp` 的源码和 `megatron.core` 在同一个 monorepo 里,单独引用需要克隆整个 Megatron-LM。
本仓把**只属于 FSDP 的部分**摘出来,可以直接 `pip install -e .`,不引入任何 Megatron-Core 依赖。

它本来就可以脱离 mcore 跑:包里每一处 `megatron.core` 引用都包在 `try/except ImportError` 里,
失败时回退到包内自带实现(`distributed_data_parallel_config.py` 是 mcore 同名类的完整副本,
`utils.py` 自带 `is_submodule` / `get_cuda_rng_tracker`)。详见 [docs/DESIGN.md](docs/DESIGN.md)。

## 安装

```bash
pip install -e .            # 必需依赖只有 torch / einops / packaging
pip install -e '.[emerging-optimizers]'   # 可选:Muon 优化器路径
```

TransformerEngine(FP8/量化)、apex(NCCL allocator)都是可选的,缺失时自动回退并只打一条警告。

## 用法

```python
import torch, torch.distributed as dist
from megatron_fsdp import fully_shard_model, fully_shard_optimizer

dist.init_process_group()
rank = dist.get_rank()
torch.cuda.set_device(rank)

model = MyModel().to(f"cuda:{rank}")
model = fully_shard_model(model, fsdp_unit_modules=[MyDecoderLayer])

optimizer = fully_shard_optimizer(torch.optim.AdamW(model.parameters(), lr=1e-4))
```

`fully_shard_model` 不传 `device_mesh` 时会自己建一个 `("fsdp", "tp")` mesh(写死 `device_type="cuda"`)。
自己管 mesh 时要注意 **`dp_shard_dim` 必须显式传**,默认值只在自动建 mesh 的分支里生效:

```python
mesh = init_device_mesh("cuda", (world, 1), mesh_dim_names=("fsdp", "tp"))
fully_shard_model(model, device_mesh=mesh, dp_shard_dim="fsdp", tp_dim="tp",
                  fsdp_unit_modules=[MyDecoderLayer])
```

全部参数(HSDP/HFSDP、EP、双缓冲、NCCL user buffer、混合精度策略等)见
[docs/UPSTREAM_API.md](docs/UPSTREAM_API.md);设计与调度原理见 [docs/mfsdp_design.md](docs/mfsdp_design.md)。

## 验证

```bash
python tests/smoke_no_mcore.py    # 导入隔离:确认 megatron 不可用、7 个公共 API 全部导出
GLOO_SOCKET_IFNAME=lo0 python -m torch.distributed.run --nproc_per_node=2 \
    tests/smoke_no_mcore.py       # 另加一次真实分片训练步(macOS 上是 lo0;需要 GPU)
```

没 GPU 时训练步会明确报 `SKIP`,不会被算成通过:`MegatronFSDP.__init__` 无条件调用
`torch.cuda.current_device()`,所以本库在纯 CPU 主机上跑不了训练步,导入是可以的。

## 这个副本不包含什么

- `mcore_fsdp_adapter.py`:上游 mcore 侧的 `FullyShardedDataParallel` 适配层,它本身依赖 mcore。
- 上游单测 `tests/unit_tests/distributed/mfsdp_{v1,v2}`:多数 import mcore,不属于可独立部分。
