# 上游溯源

| 项 | 值 |
|---|---|
| 上游仓 | https://github.com/NVIDIA/Megatron-LM |
| 镜像 commit | `dc36b1a89f5de29b09875a03997fe6f1334e2a73`(2026-10-28,`ci(sync): remove scheduled main-to-dev workflow (#8603)`) |
| 包目录最后改动 | `47715bb029f59aef4c45d516379911729e6b375d`(2026-10-27,`Add unpadded shards for MFSDP v2 RowAtomic/BlockAtomic layouts (#8116)`) |
| 上游路径 | `megatron/core/distributed/fsdp/src/` |
| 包版本 | `megatron_fsdp` 0.8.0rc0(`megatron_fsdp/package_info.py`) |
| PyPI | https://pypi.org/project/megatron-fsdp/ |
| 层次 | 15195 行 Python / 29 个模块(含 `experimental/` 22 个) |

## 同步方式

本仓文件与上游 `src/` 逐字节一致,除以下呈现层改动:

| 文件 | 改动 |
|---|---|
| `README.md` | 重写为本仓的使用说明(上游原文移到 `docs/UPSTREAM_API.md`) |
| `pyproject.toml` | 保留上游配置,仅改 readme 指向并补充元数据 |
| `docs/DESIGN.md` | 新增,记录剥离边界 |
| `docs/UPSTREAM_API.md` | 上游 `src/README.md` 原样 |
| `tests/smoke_no_mcore.py` | 新增,验证无 mcore 可导入 |
| `LICENSE` | 从上游仓根复制 |

Python 源码(`megatron_fsdp/**`)未做任何修改。

## 许可

上游仓根 `LICENSE` 是复合许可:默认 BSD-3,内嵌 Apache License 2.0(第 42 行起)与 BSD 条款(第 302 行起)。
包内各文件头声明 Apache-2.0,已在 LICENSE 覆盖范围内。本仓原样保留全部版权声明,未添加本仓署名。
