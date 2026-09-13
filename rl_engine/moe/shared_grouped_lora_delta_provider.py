from __future__ import annotations

from typing import Any

import torch

#from rl_engine.moe.contract import ORACLE_PROFILE
from rl_engine.moe.provider import ReferenceProvider

class LoRADeltaProvider(ReferenceProvider):
    """P5-3 (#62): shared_grouped_lora_delta 的 CUDA/Triton 实现。

    只覆盖这一个算子的 fwd/bwd, 其余 8 个方法继承自 ReferenceProvider,
    继续走 FP32 oracle —— 这样整条验收流水线从第一天就能跑通。
    """

    # name / numeric_profile: provider 的身份标签, 会打印在验收输出的最后一行,
    # 形如 provider=p5-3-lora-delta profile=... device=cpu
    name = "p5-3-lora-delta"
    numeric_profile = "torch-native-nondeterministic"

    # provenance = "出处": 声明这次到底用什么后端跑的。
    # 返回值要贴进 PR 描述作为证据 (start kit 第 80 行要求)。
    def provenance(self) -> dict[str, Any]:
        return {
            "requested_backend": self.name,
            "actual_backend": "torch-native",
            "numeric_profile": self.numeric_profile,
            "torch_version": torch.__version__,
        }

    def shared_grouped_lora_delta_fwd(
        self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 输入的x,a,b都是bf16，返回值除了u_bf16则都是fp32
        # 第 1 步: 先原样转调 oracle, 确认接线正确、验收能跑绿
        # 第 2 步: 换成你自己的 torch 实现
        # 第 3/4 步: 换成 CUDA / Triton
        """Native BF16 LoRA delta: Y = (X @ A.T) @ B.T * alpha, FP32 accumulate."""
        # 问: 输入 x/a 都是 BF16, 为何 u 是 FP32? 是 @ 自带隐式转换吗?
        # 答: 不是。@ 的输出 dtype 等于输入 dtype (bf16@bf16 -> bf16),
        #     两边 dtype 不同还会直接报错。是 .float() 把它们转成了 FP32。
        u_fp32 = x.float() @ a.float().T  # [M, r] FP32
        u_bf16 = u_fp32.to(torch.bfloat16)  # 舍入点1
        y_fp32 = (u_bf16.float() @ b.float().T) * float(alpha)  # [M, N] FP32
        return y_fp32, u_bf16

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

        """Returns (dX, dA, dB) as FP32. No dW — base weights are frozen."""
        # dtype 规则 (契约定死的):
        #   输入永远是 BF16 -> 算之前两边都 .float(), 在 FP32 里累加
        #   只有 3 个舍入点回 BF16: fwd 的 u, bwd 的 dys 和 du
        #   返回值全是 FP32 (契约 D7: 梯度 = 累加器 dtype)
        # x.float() 返回一个新的 FP32 张量, x 本身不变
        '''初始版本
        dys = (dy.to(torch.float32) * float(alpha)).to(torch.bfloat16)
        du = dys @ b
        du_bf16 = du.to(torch.bfloat16)
        db = dys.t @ u_bf16
        da = du_bf16.t @ x
        dx = du_bf16 @ a
        return dx, da, db
        '''
        dys = (dy.float() * float(alpha)).to(torch.bfloat16)  # 舍入点2
        du_fp32 = dys.float() @ b.float()  # [M, r] FP32
        du_bf16 = du_fp32.to(torch.bfloat16)  # [M, r] BF16
        db = dys.float().T @ u_bf16.float()  # [N, r] FP32
        da = du_bf16.float().T @ x.float()  # [r, K] FP32
        dx = du_bf16.float() @ a.float()  # [M, K] FP32
        return dx, da, db
