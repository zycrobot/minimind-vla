"""
简单的 VLA 模型测试脚本
直接运行即可测试模型并打印 action
"""
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from model.model_vla import MiniMindVLA, VLAConfig
from model.model_vlm import MiniMindVLM
from dataset.vla_dataloader import VLADataset

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 创建 VLA 配置
config = VLAConfig(
    hidden_size=512,
    num_hidden_layers=8,
    action_dim=8,  # 8维动作（例如：7DOF机械臂 + gripper）
    action_chunk_size=100,  # 预测16步动作
    action_hidden_size=256,
)

# 初始化模型
print("正在初始化 VLA 模型...")
model = MiniMindVLA(
    params=config,
    vision_model_path="./model/vision_model/clip-vit-base-patch16"
)
model = model.eval().to(device)
print("模型初始化完成")

# 加载 tokenizer
print("正在加载 tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("model")
    print("Tokenizer 加载完成")
except:
    print("警告: 无法加载 tokenizer，使用默认配置")
    tokenizer = None

# 准备测试数据
print("\n准备测试数据...")

# 尝试从数据集中加载真实数据
use_dataset = True
pixel_values = None
robot_states = None

if use_dataset:
    try:
        # 替换为实际的数据集路径
        dataset_path = "e:\\project\\minimind\\minimind-v\\dataset\\franka_pick_dataset.hdf5"
        print(f"尝试从数据集加载数据: {dataset_path}")
        
        # 创建数据集实例
        dataset = VLADataset(
            hdf5_path=dataset_path,
            data_group='data',
            tokenizer=None,
            max_length=512,
            action_chunk_size=config.action_chunk_size,
            action_dim=config.action_dim,
            robot_state_dim=config.robot_state_dim
        )
        
        # 确保数据集不为空
        if len(dataset) > 0:
            print(f"数据集加载成功，包含 {len(dataset)} 个样本")
            print(f"包含机器人状态: {dataset.has_robot_state}")
            
            # 获取第一个样本
            _, _, _, image_tensors, _, robot_state_tensor = dataset[0]
            
            # 处理图像和机器人状态
            pixel_values = image_tensors.to(device)
            robot_states = robot_state_tensor.unsqueeze(0).to(device)  # 添加batch维度
            
            print(f"成功加载图像: {pixel_values.shape}")
            print(f"成功加载机器人状态: {robot_states.shape}")
            print(f"机器人状态值: {robot_states.cpu().numpy()}")
        else:
            print("数据集为空，使用默认数据")
            use_dataset = False
    except Exception as e:
        print(f"加载数据集失败: {e}，使用默认数据")
        use_dataset = False

# 如果无法从数据集加载，使用默认数据
if not use_dataset:
    # 创建测试图像（224x224 RGB）
    test_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
    print("创建测试图像: 224x224 灰色图像")
    
    # 处理图像
    if model.processor is not None:
        pixel_values = MiniMindVLM.image2tensor(test_image, model.processor).to(device).unsqueeze(0)
        print("图像处理完成")
    else:
        print("警告: 无法处理图像，使用 None")
        pixel_values = None
    
    # 创建机器人状态（8维：7个关节角度 + 1个夹爪值）
    robot_states = torch.zeros(1, config.robot_state_dim, device=device)
    print(f"创建机器人状态: {robot_states.shape} (8维：7个关节角度 + 1个夹爪值)")
    print(f"机器人状态值: {robot_states.cpu().numpy()}")

# 准备文本输入
prompt = "请执行抓取动作"
print(f"文本提示: {prompt}")

if tokenizer is not None:
    messages = [{"role": "user", "content": prompt}]
    try:
        inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(device)
        print("文本处理完成")
    except:
        # 如果 apply_chat_template 失败，直接编码
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        print("使用简单文本编码")
else:
    # 如果没有 tokenizer，创建虚拟输入
    print("警告: 没有 tokenizer，使用虚拟输入")
    inputs = {"input_ids": torch.randint(0, 1000, (1, 10)).to(device), 
              "attention_mask": torch.ones(1, 10).to(device)}

# 运行推理
print("\n" + "="*80)
print("开始推理...")
print("="*80)

with torch.no_grad():
    output = model.forward(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=pixel_values,
        robot_states=robot_states,
        return_actions=True,
    )

# 获取预测的 action
predicted_actions = output.get('actions', None)

# 打印结果
print("\n" + "="*80)
print("推理结果")
print("="*80)

if predicted_actions is not None:
    actions_np = predicted_actions.cpu().numpy()
    
    # 如果是批量数据，只取第一个
    if len(actions_np.shape) == 3:
        actions_np = actions_np[0]
    
    print(f"\n预测的 Action Chunk 形状: {actions_np.shape}")
    print(f"  - Chunk Size: {actions_np.shape[0]}")
    print(f"  - Action Dim: {actions_np.shape[1]}")
    
    print("\n详细的 Action 序列:")
    print("-"*80)
    for i, action_step in enumerate(actions_np):
        print(f"Step {i+1:2d}: ", end="")
        action_str = ", ".join([f"{val:8.4f}" for val in action_step])
        print(f"[{action_str}]")
    
    print("-"*80)
    print("\nAction 统计信息:")
    print(f"  均值: {np.mean(actions_np, axis=0)}")
    print(f"  标准差: {np.std(actions_np, axis=0)}")
    print(f"  最小值: {np.min(actions_np, axis=0)}")
    print(f"  最大值: {np.max(actions_np, axis=0)}")
    
    # 也可以使用便捷方法
    print("\n" + "="*80)
    print("使用 predict_actions 方法:")
    print("="*80)
    predicted_actions_2 = model.predict_actions(
        input_ids=inputs["input_ids"],
        pixel_values=pixel_values,
        robot_states=robot_states,
    )
    
    if predicted_actions_2 is not None:
        actions_np_2 = predicted_actions_2.cpu().numpy()
        if len(actions_np_2.shape) == 3:
            actions_np_2 = actions_np_2[0]
        print(f"预测的 Action Chunk 形状: {actions_np_2.shape}")
        print("前3步动作:")
        for i in range(min(3, len(actions_np_2))):
            print(f"  Step {i+1}: {actions_np_2[i]}")
else:
    print("警告: 模型未返回 action")

print("\n" + "="*80)
print("测试完成！")
print("="*80)

