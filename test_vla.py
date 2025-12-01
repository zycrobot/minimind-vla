import argparse
import os
import warnings
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from model.model_vla import MiniMindVLA, VLAConfig
from model.model_vlm import MiniMindVLM
from trainer.trainer_utils import setup_seed

warnings.filterwarnings('ignore')


def init_model(args):
    """初始化 VLA 模型"""
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    
    # 创建 VLA 配置
    vla_config = VLAConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        action_dim=args.action_dim,
        action_chunk_size=args.action_chunk_size,
        action_hidden_size=args.action_hidden_size,
    )
    
    # 初始化模型
    model = MiniMindVLA(
        params=vla_config,
        vision_model_path="./model/vision_model/clip-vit-base-patch16"
    )
    
    # 如果提供了权重路径，加载权重
    if args.load_weight and os.path.exists(args.load_weight):
        print(f"正在加载权重: {args.load_weight}")
        state_dict = torch.load(args.load_weight, map_location=args.device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
        print("权重加载完成")
    else:
        print("使用随机初始化的模型权重")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'VLA模型总参数量: {total_params / 1e6:.2f} M')
    print(f'VLA模型可训练参数量: {trainable_params / 1e6:.2f} M')
    
    preprocess = model.processor
    return model.eval().to(args.device), tokenizer, preprocess


def print_actions(actions, action_dim, action_chunk_size):
    """格式化打印 action"""
    print("\n" + "="*80)
    print("预测的 Action Chunk:")
    print("="*80)
    
    if actions is None:
        print("未生成 action")
        return
    
    actions_np = actions.cpu().numpy() if torch.is_tensor(actions) else actions
    
    # 如果是批量数据，只打印第一个
    if len(actions_np.shape) == 3:
        actions_np = actions_np[0]
    
    print(f"Action 形状: {actions_np.shape} (chunk_size={action_chunk_size}, action_dim={action_dim})")
    print("\n详细 Action 序列:")
    print("-"*80)
    
    for i, action_step in enumerate(actions_np):
        print(f"Step {i+1:2d}: ", end="")
        action_str = ", ".join([f"{val:8.4f}" for val in action_step])
        print(action_str)
    
    print("-"*80)
    print(f"Action 统计信息:")
    print(f"  均值: {np.mean(actions_np, axis=0)}")
    print(f"  标准差: {np.std(actions_np, axis=0)}")
    print(f"  最小值: {np.min(actions_np, axis=0)}")
    print(f"  最大值: {np.max(actions_np, axis=0)}")
    print("="*80 + "\n")


def test_vla_model(args):
    """测试 VLA 模型"""
    # 初始化模型
    model, tokenizer, preprocess = init_model(args)
    
    # 设置随机种子
    setup_seed(args.seed)
    
    # 准备测试数据
    if args.image_path and os.path.exists(args.image_path):
        # 从文件加载图像
        image = Image.open(args.image_path).convert('RGB')
        print(f"加载图像: {args.image_path}")
    else:
        # 创建随机测试图像
        print("使用随机生成的测试图像")
        image = Image.new('RGB', (224, 224), color=(128, 128, 128))
    
    # 准备文本提示
    prompt = args.prompt if args.prompt else "请执行抓取动作"
    print(f"文本提示: {prompt}")
    
    # 处理图像
    pixel_values = MiniMindVLM.image2tensor(image, preprocess).to(args.device).unsqueeze(0)
    
    # 准备文本输入
    messages = [{"role": "user", "content": prompt}]
    inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(args.device)
    
    print("\n开始推理...")
    print("-"*80)
    
    # 运行模型推理
    with torch.no_grad():
        output = model.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=pixel_values,
            return_actions=True,
        )
    
    # 获取预测的 action
    predicted_actions = output.get('actions', None)
    
    # 打印 action
    if predicted_actions is not None:
        print_actions(
            predicted_actions,
            args.action_dim,
            args.action_chunk_size
        )
    else:
        print("警告: 模型未返回 action")
    
    # 也可以使用便捷方法
    print("\n使用 predict_actions 方法:")
    print("-"*80)
    predicted_actions_2 = model.predict_actions(
        input_ids=inputs["input_ids"],
        pixel_values=pixel_values,
    )
    
    if predicted_actions_2 is not None:
        print_actions(
            predicted_actions_2,
            args.action_dim,
            args.action_chunk_size
        )


def main():
    parser = argparse.ArgumentParser(description="测试 MiniMind-VLA 模型并打印 Action")
    
    # 模型配置参数
    parser.add_argument('--load_from', default='model', type=str, 
                       help="tokenizer 加载路径")
    parser.add_argument('--load_weight', default='', type=str, 
                       help="模型权重路径（可选，如果提供则加载权重）")
    parser.add_argument('--hidden_size', default=512, type=int, 
                       help="隐藏层维度（512=Small-26M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, 
                       help="隐藏层数量（Small=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], 
                       help="是否使用MoE架构（0=否，1=是）")
    
    # Action 配置参数
    parser.add_argument('--action_dim', default=7, type=int, 
                       help="动作维度（例如：6DOF机械臂 + gripper）")
    parser.add_argument('--action_chunk_size', default=16, type=int, 
                       help="action chunk 的大小")
    parser.add_argument('--action_hidden_size', default=256, type=int, 
                       help="action 模块的隐藏层大小")
    
    # 测试参数
    parser.add_argument('--image_path', default='', type=str, 
                       help="测试图像路径（可选，如果不提供则使用随机图像）")
    parser.add_argument('--prompt', default='', type=str, 
                       help="文本提示（可选，默认：'请执行抓取动作'）")
    parser.add_argument('--seed', default=2026, type=int, 
                       help="随机种子")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                       type=str, help="运行设备")
    
    args = parser.parse_args()
    
    print("="*80)
    print("MiniMind-VLA 模型测试")
    print("="*80)
    print(f"设备: {args.device}")
    print(f"Action 维度: {args.action_dim}")
    print(f"Action Chunk 大小: {args.action_chunk_size}")
    print("="*80)
    
    test_vla_model(args)
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()

