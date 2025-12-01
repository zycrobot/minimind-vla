"""
在 PyBullet 仿真环境中测试已经训练好的 MiniMind-VLA 模型。

功能：
1. 加载 Franka 仿真环境（可视化可选）
2. 对随机放置的多种物体生成“请抓取XX”指令
3. 调用 VLA 模型生成 action chunk，并按顺序驱动机械臂
4. 可选地将渲染画面保存为 mp4 视频
"""
import argparse
import os
from pathlib import Path
from typing import List

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import imageio
import numpy as np
import pybullet as pb
import torch
from PIL import Image
from transformers import AutoTokenizer

from model.model_vla import MiniMindVLA, VLAConfig
from model.model_vlm import MiniMindVLM
from scripts.collect_franka_hdf5 import FrankaPyBulletEnv


def prepare_inputs(tokenizer, prompt: str, device: torch.device):
    messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
    try:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted = prompt
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    return {k: v.to(device) for k, v in inputs.items()}


@torch.no_grad()
def infer_actions(model: MiniMindVLA, tokenizer, image_np: np.ndarray, robot_state: np.ndarray, prompt: str, device: torch.device):
    pil_img = Image.fromarray(image_np)
    pixel_tensor = MiniMindVLM.image2tensor(pil_img, model.processor).unsqueeze(0).to(device)
    inputs = prepare_inputs(tokenizer, prompt, device)
    # 转换机器人状态为张量并添加批次维度
    robot_state_tensor = torch.tensor(robot_state, dtype=torch.float32).unsqueeze(0).to(device)
    actions = model.predict_actions(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        pixel_values=pixel_tensor,
        robot_states=robot_state_tensor,
    )
    return actions[0].detach().cpu().numpy()

def smooth_actions(actions, window_size=3):
    """使用移动平均平滑动作序列"""
    if len(actions) < window_size:
        return actions
    
    smoothed_actions = np.copy(actions)
    # 对每个关节分别进行平滑处理
    for joint_idx in range(actions.shape[1]):
        for i in range(1, len(actions) - 1):
            # 简单的三点移动平均
            smoothed_actions[i, joint_idx] = np.mean(actions[max(0, i-1):min(len(actions), i+2), joint_idx])
    
    return smoothed_actions

def get_robot_state(env: FrankaPyBulletEnv) -> np.ndarray:
    """
    从环境中获取机器人状态
    返回：[7个机械臂关节角度, 1个夹爪关节角度]
    与collect_franka_hdf5.py中的实现保持一致
    """
    # 使用环境自带的get_robot_state方法，确保一致性
    return env.get_robot_state()


def save_video(frames: List[np.ndarray], out_path: Path, fps: int = 30):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def main():
    parser = argparse.ArgumentParser(description="MiniMind-VLA 仿真评估")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/post_train_vla_512.pth", help="模型权重路径")
    parser.add_argument("--from_weight", type=str, default="pretrain_vlm", help="若需要先加载预训练权重，可指定")
    parser.add_argument("--episodes", type=int, default=5, help="测试轮数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gui", action="store_true", help="是否展示 PyBullet GUI")
    parser.add_argument("--video_dir", type=str, default="videos", help="保存测试视频的目录（为空则不保存）")
    parser.add_argument("--steps_per_action", type=int, default=30, help="执行每个动作块的仿真步数")
    parser.add_argument("--smooth_actions", action="store_true", default=True, help="是否启用动作平滑处理")
    parser.add_argument("--smoothing_window", type=int, default=3, help="动作平滑的窗口大小")
    parser.add_argument("--position_gain", type=float, default=0.03, help="位置控制增益参数，减小可降低抖动")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. 初始化模型
    vla_config = VLAConfig(
        hidden_size=512,
        num_hidden_layers=8,
        action_dim=8,  # 7个机械臂关节 + 1个夹爪值
        action_chunk_size=100,
        action_hidden_size=256,
        robot_state_dim=8,  # 7个机械臂关节 + 1个夹爪值
    )
    tokenizer = AutoTokenizer.from_pretrained("model")
    model = MiniMindVLA(vla_config).to(device)
    model.eval()

    if args.from_weight and args.from_weight != "none":
        pretrain_path = f'out/{args.from_weight}_{vla_config.hidden_size}.pth'


        
        if os.path.exists(pretrain_path):
            state = torch.load(pretrain_path, map_location=device)
            model.load_state_dict(state, strict=False)

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # 处理权重维度不匹配问题
        model_dict = model.state_dict()
        filtered_checkpoint = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered_checkpoint[k] = v
                else:
                    print(f"跳过维度不匹配的参数: {k}, 检查点形状: {v.shape}, 模型形状: {model_dict[k].shape}")
                    # 对于velocity_net，我们只复制可兼容的部分权重
                    if 'velocity_net.0.weight' in k:
                        # 获取兼容的列数
                        compatible_cols = min(v.shape[1], model_dict[k].shape[1])
                        # 初始化新权重
                        new_weight = torch.zeros_like(model_dict[k])
                        # 复制兼容部分
                        new_weight[:, :compatible_cols] = v[:, :compatible_cols]
                        # 初始化新增加的部分（使用 Xavier 初始化）
                        torch.nn.init.xavier_uniform_(new_weight[:, compatible_cols:])
                        filtered_checkpoint[k] = new_weight
                    elif 'velocity_net.0.bias' in k:
                        # 对于偏置，直接使用
                        filtered_checkpoint[k] = v
        # 加载过滤后的权重
        model.load_state_dict(filtered_checkpoint, strict=False)
        print(f"已加载检查点: {args.checkpoint}")
    else:
        raise FileNotFoundError(f"未找到模型权重: {args.checkpoint}")

    # 2. 初始化仿真环境
    env = FrankaPyBulletEnv(gui=args.gui)

    # 3. 逐个 episode 测试
    for ep in range(args.episodes):
        env.reset_robot()
        env.reset_objects()
        env.spawn_objects()
        target_name = np.random.choice(list(env.object_ids.keys()))
        instruction = f"请抓取{target_name}"
        print(f"[Episode {ep + 1}] 指令: {instruction}")

        frames = []
        observation = env.render()
        frames.append(observation)

        # 获取机器人当前状态
        robot_state = get_robot_state(env)
        action_seq = infer_actions(model, tokenizer, observation, robot_state, instruction, device)
        
        # 启用动作平滑处理
        if args.smooth_actions:
            print(f"应用动作平滑处理，窗口大小: {args.smoothing_window}")
            action_seq = smooth_actions(action_seq, args.smoothing_window)
        
        # 获取当前机械臂和夹爪状态作为初始位置，用于平滑过渡
        current_full_state = env.get_robot_state()  # 8维状态
        current_joint_positions = current_full_state[:7]  # 只取机械臂部分
        current_gripper_value = current_full_state[7]  # 取夹爪部分
        
        # 第一个动作与当前位置之间添加平滑过渡
        if len(action_seq) > 0:
            transition_steps = 15  # 过渡步数
            first_action = action_seq[0]
            for t in range(transition_steps):
                # 线性插值计算过渡动作
                alpha = t / transition_steps
                # 对机械臂部分进行插值
                transition_arm = (1 - alpha) * current_joint_positions + alpha * first_action[:7]
                # 对夹爪部分进行插值
                transition_gripper = (1 - alpha) * current_gripper_value + alpha * first_action[7]
                # 组合过渡动作
                transition_action = np.concatenate([transition_arm, [transition_gripper]])
                # 应用过渡动作，只使用前7个机械臂关节值
                env.apply_joint_positions(transition_action[:7], steps=1, position_gain=args.position_gain)
                # 同时控制夹爪
                pb.setJointMotorControl2(
                    env.robot_id,
                    env.gripper_joint_index,
                    pb.POSITION_CONTROL,
                    targetPosition=float(transition_gripper),
                    force=100,
                )
                pb.setJointMotorControl2(
                    env.robot_id,
                    10,  # 另一个夹爪关节
                    pb.POSITION_CONTROL,
                    targetPosition=float(transition_gripper),
                    force=100,
                )
                pb.stepSimulation()
                frames.append(env.render())
        
        # 执行剩余的动作序列
        for action in action_seq[1:]:
            # 应用机械臂部分的动作
            env.apply_joint_positions(action[:7], steps=args.steps_per_action, position_gain=args.position_gain)
            # 同时控制夹爪
            pb.setJointMotorControl2(
                env.robot_id,
                env.gripper_joint_index,
                pb.POSITION_CONTROL,
                targetPosition=float(action[7]),
                force=100,
            )
            pb.setJointMotorControl2(
                env.robot_id,
                10,  # 另一个夹爪关节
                pb.POSITION_CONTROL,
                targetPosition=float(action[7]),
                force=100,
            )
            for _ in range(args.steps_per_action):
                pb.stepSimulation()
            frames.append(env.render())

        env.close_gripper()
        for _ in range(60):
            pb.stepSimulation()
            frames.append(env.render())
        env.move_to(env.workspace_center + np.array([0, 0, 0.4]), pb.getQuaternionFromEuler([0, np.pi, 0]))
        frames.append(env.render())

        if args.video_dir:
            out_path = Path(args.video_dir) / f"episode_{ep + 1:02d}.mp4"
            save_video(frames, out_path)
            print(f"保存视频: {out_path}")

    print("评估完成！")


if __name__ == "__main__":
    main()

