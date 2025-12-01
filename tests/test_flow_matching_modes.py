"""
简单测试 FlowMatchingAction 在 x/eps/v 三种预测模式下是否能正常前向与采样。
运行该脚本无需加载完整模型或权重，适合作为快速回归测试。
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.model_vla import FlowMatchingAction


def run_case(target_type: str) -> None:
    batch_size = 2
    seq_len = 4
    hidden_size = 32
    action_dim = 3
    chunk_size = 4

    head = FlowMatchingAction(
        hidden_size=hidden_size,
        action_dim=action_dim,
        action_chunk_size=chunk_size,
        action_hidden_size=64,
        num_layers=2,
        target_type=target_type,
        flow_matching_sigma=0.1,
        num_sampling_steps=10,
    )

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    action_targets = torch.randn(batch_size, chunk_size, action_dim)

    head.train()
    predicted, loss = head(
        hidden_states=hidden_states,
        action_targets=action_targets,
        training=True,
    )

    assert predicted.shape == (batch_size, chunk_size, action_dim)
    print(f"[{target_type}] train loss = {loss.item():.6f}")

    head.eval()
    sampled_actions = head(
        hidden_states=hidden_states,
        action_targets=None,
        training=False,
    )
    assert sampled_actions.shape == (batch_size, chunk_size, action_dim)
    print(f"[{target_type}] inference sample mean = {sampled_actions.mean().item():.6f}")


if __name__ == "__main__":
    for mode in ("x", "eps", "v"):
        run_case(mode)

