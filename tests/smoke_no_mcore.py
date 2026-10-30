"""Prove megatron_fsdp works with megatron.core absent.

Two independent checks, both required to pass for the vendored copy to be
considered usable standalone:

  1. import isolation — `megatron` is not importable, yet every public symbol
     from `megatron_fsdp` loads;
  2. a real sharded training step — fully_shard_model + fully_shard_optimizer
     over a 2-rank gloo mesh, one forward/backward/step, with the optimizer
     loss decreasing and every rank agreeing on the final parameter values.

Run (needs 2 GPUs; GLOO_SOCKET_IFNAME selects the loopback interface — on
macOS that is lo0, not lo):

    GLOO_SOCKET_IFNAME=lo0 python -m torch.distributed.run --nproc_per_node=2 \
        --master_port=29517 tests/smoke_no_mcore.py

On a host without CUDA the import checks still run and the training step is
reported as SKIPPED, never as passed: MegatronFSDP.__init__ calls
torch.cuda.current_device() unconditionally (megatron_fsdp.py:240), so the
library cannot execute a step on a CPU-only host even with device="cpu".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Import the vendored package from this repo, not from site-packages, whether or
# not it is installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}", flush=True)
    if not cond:
        FAILURES.append(label)


def check_import_isolation() -> None:
    try:
        import megatron  # noqa: F401

        check(False, "megatron.core is absent (it was importable!)")
    except ImportError:
        check(True, "megatron.core is absent")

    import megatron_fsdp

    check("megatron_fsdp/" in megatron_fsdp.__file__, "megatron_fsdp imported from this repo")
    expected = {
        "DistributedDataParallelConfig",
        "MegatronFSDP",
        "FSDPDistributedIndex",
        "MixedPrecisionPolicy",
        "fully_shard",
        "fully_shard_model",
        "fully_shard_optimizer",
    }
    check(expected <= set(megatron_fsdp.__all__), "all 7 public symbols exported")
    check(megatron_fsdp.__version__ == "0.8.0rc0", "version is 0.8.0rc0")


class TinyBlock(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).relu()


class TinyModel(torch.nn.Module):
    def __init__(self, dim: int, depth: int) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([TinyBlock(dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def check_training_step() -> None:
    from torch.distributed.device_mesh import init_device_mesh

    from megatron_fsdp import fully_shard_model, fully_shard_optimizer

    world = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    # dp_shard_dim defaults to "fsdp" only on the auto-mesh path; with an
    # explicit mesh it must be named or FSDPDistributedIndex rejects it.
    mesh = init_device_mesh("cuda", (world, 1), mesh_dim_names=("fsdp", "tp"))

    torch.manual_seed(0)  # identical init on every rank before sharding
    model = TinyModel(dim=8, depth=4).to(f"cuda:{rank}")
    model = fully_shard_model(
        model,
        device_mesh=mesh,
        dp_shard_dim="fsdp",
        tp_dim="tp",
        fsdp_unit_modules=[TinyBlock],
        device=torch.device(f"cuda:{rank}"),
    )

    opt = fully_shard_optimizer(torch.optim.AdamW(model.parameters(), lr=1e-2))
    check(opt is not None, "fully_shard_optimizer returned an optimizer")

    x = torch.ones(4, 8, device=f"cuda:{rank}")
    losses: list[float] = []
    for _ in range(3):
        opt.zero_grad()
        loss = model(x).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    check(losses[-1] < losses[0], f"loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # Every rank must materialise the same weights after the step.
    flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    gathered = [torch.empty_like(flat) for _ in range(world)]
    dist.all_gather(gathered, flat)
    same = all(torch.equal(gathered[0], g) for g in gathered[1:])
    check(same, f"all {world} ranks agree on final parameters")

    if rank == 0:
        print(f"  param count={flat.numel()} losses={[round(v, 5) for v in losses]}")


def main() -> int:
    check_import_isolation()

    if "RANK" not in os.environ:
        print("[SKIP] training step (launch with torch.distributed.run to run it)", flush=True)
    elif not torch.cuda.is_available():
        print(
            "[SKIP] training step (no CUDA: MegatronFSDP.__init__ calls "
            "torch.cuda.current_device() unconditionally — not a pass)",
            flush=True,
        )
    else:
        dist.init_process_group(backend="nccl")
        try:
            check_training_step()
        finally:
            dist.destroy_process_group()

    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) FAILED: {FAILURES}", flush=True)
        return 1
    print("\nAll checks passed — megatron_fsdp runs without megatron.core.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
