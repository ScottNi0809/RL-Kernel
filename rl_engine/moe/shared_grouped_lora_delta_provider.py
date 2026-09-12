from __future__ import annotations

from typing import Any

import torch

from rl_engine.moe import oracle
from rl_engine.moe.contract import ORACLE_PROFILE
from rl_engine.moe.provider import ReferenceProvider

class LoRADeltaProvider(ReferenceProvider):
    """P5-3 (#62): shared_grouped_lora_delta 的 CUDA/Triton 实现。

    只覆盖这一个算子的 fwd/bwd, 其余 8 个方法继承自 ReferenceProvider,
    继续走 FP32 oracle —— 这样整条验收流水线从第一天就能跑通。
    """

    name = "p5-3-lora-delta" # provider 的身份标签，会打印在验收输出的最后一行
    numeric_profile = ORACLE_PROFILE # provider=p5-3-lora-delta  profile=oracle-fp32-serial-v1  device=cpu

    def provenance(self) -> dict[str, Any]: # 这个方法和方法名是什么意思？ - 说这个方法的返回值要贴进 PR 描述，作为证据
        return {
            "requested_backend": self.name,
            "actual_backend": "oracle_passthr",  # 填真实现后改成 "cuda" / "triton"
            "numeric_profile": self.numeric_profile,
            "torch_version": torch.__version__,
        }

    def shared_grouped_lora_delta_fwd(
        self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 第 1 步: 先原样转调 oracle, 确认接线正确、验收能跑绿
        # 第 2 步: 换成你自己的 torch 实现
        # 第 3/4 步: 换成 CUDA / Triton
        return oracle.shared_grouped_lora_delta_fwd(x, a, b, alpha)

    def shared_grouped_lora_delta_bwd(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        alpha: float,
        u_bf16: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward of the same graph: returns ``(dX, dA, dB)`` as FP32.

        ``dY' = dY * alpha`` rounds to BF16, then each GEMM runs BF16-in /
        FP32-accumulate, serial ascending order; ``dU`` rounds to BF16 before
        reuse. Association order is frozen as written.
        """
        return oracle.shared_grouped_lora_delta_bwd(dy, x, a, b, alpha, u_bf16)