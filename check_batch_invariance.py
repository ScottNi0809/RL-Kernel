"""P5-3 batch-invariance probe.

A row's output must be bit-identical whether it is computed alone or inside a
larger batch. torch.matmul does not guarantee this: cuBLAS picks tile shapes and
split-k strategies from the matrix dimensions, so the accumulation order changes
with M.

The naive probe (single seed, only the final output) is misleading -- the BF16
rounding of the intermediate `u` absorbs most FP32 drift, so a run can look clean
while the underlying GEMM is already non-deterministic. This script therefore:

  * checks every boundary, not just the final output;
  * sweeps several seeds and shapes, since a mismatch only surfaces when a value
    lands near a BF16 rounding boundary;
  * reports how many elements drifted, so a "pass" can be told apart from a
    "passed by luck".
"""

from __future__ import annotations

import torch

from rl_engine.moe.shared_grouped_lora_delta_provider import LoRADeltaProvider

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (0, 1, 2, 42, 2026)
SUB_BATCHES = (1, 2, 3, 8, 17)

# (M, K, N, r) -- the last one matches the P5 fixture geometry.
SHAPES = (
    (24, 128, 64, 8),
    (32, 256, 128, 16),
    (24, 128, 128, 8),
)
ALPHA = 0.5


def _drift(sub: torch.Tensor, ref: torch.Tensor) -> tuple[int, float]:
    """Element count that differs, and the largest absolute difference."""
    diff = (sub.float() - ref.float()).abs()
    return int((diff > 0).sum().item()), float(diff.max().item())


def probe_one(provider, m_full: int, k: int, n: int, r: int, seed: int) -> list[dict]:
    torch.manual_seed(seed)
    x = torch.randn(m_full, k, dtype=torch.bfloat16, device=DEVICE)
    a = torch.randn(r, k, dtype=torch.bfloat16, device=DEVICE)
    b = torch.randn(n, r, dtype=torch.bfloat16, device=DEVICE)
    dy = torch.randn(m_full, n, dtype=torch.bfloat16, device=DEVICE)

    y_full, u_full = provider.shared_grouped_lora_delta_fwd(x, a, b, ALPHA)
    dx_full, da_full, db_full = provider.shared_grouped_lora_delta_bwd(
        dy, x, a, b, ALPHA, u_full
    )

    rows = []
    for m in SUB_BATCHES:
        if m > m_full:
            continue
        y_sub, u_sub = provider.shared_grouped_lora_delta_fwd(x[:m], a, b, ALPHA)
        dx_sub, _, _ = provider.shared_grouped_lora_delta_bwd(
            dy[:m], x[:m], a, b, ALPHA, u_sub
        )
        # dA/dB are reductions over all rows, so they are not sliceable -- only
        # the per-row outputs (y, u, dX) can be compared against the full batch.
        rows.append(
            {
                "m": m,
                "u": torch.equal(u_sub, u_full[:m]),
                "y": torch.equal(y_sub, y_full[:m]),
                "dx": torch.equal(dx_sub, dx_full[:m]),
                "u_drift": _drift(u_sub, u_full[:m]),
                "y_drift": _drift(y_sub, y_full[:m]),
            }
        )
    return rows


def main() -> int:
    provider = LoRADeltaProvider()
    backend = provider.provenance()["actual_backend"]
    print(f"device={DEVICE}  provider={provider.name}  backend={backend}")

    total = 0
    failed = 0
    for shape in SHAPES:
        m_full, k, n, r = shape
        print(f"\n=== shape M={m_full} K={k} N={n} r={r} ===")
        print(
            f"{'seed':>6} {'M':>4} {'u':>7} {'y':>7} {'dX':>7}"
            f"   {'u drift':>16}   {'y drift':>16}"
        )
        for seed in SEEDS:
            for row in probe_one(provider, m_full, k, n, r, seed):
                total += 1
                ok = row["u"] and row["y"] and row["dx"]
                failed += 0 if ok else 1
                u_cnt, u_mx = row["u_drift"]
                y_cnt, y_mx = row["y_drift"]
                print(
                    f"{seed:>6} {row['m']:>4} "
                    f"{str(row['u']):>7} {str(row['y']):>7} {str(row['dx']):>7}   "
                    f"{u_cnt:>5} / {u_mx:.2e}   {y_cnt:>5} / {y_mx:.2e}"
                )

    print(f"\n{'=' * 60}")
    print(f"checks: {total}   violations: {failed}")
    if failed:
        print(
            "RESULT: NOT batch-invariant.\n"
            "  torch.matmul changes its accumulation order with M, so a row's\n"
            "  result depends on the batch it was computed in. A hand-written\n"
            "  kernel with a fixed k-loop is required to fix this."
        )
        return 1
    print("RESULT: batch-invariant across all probed seeds and shapes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
