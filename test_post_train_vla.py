"""
测试 VLA 训练脚本
"""
import os
import sys
import subprocess
import torch

def test_vla_training():
    """测试 VLA 训练流程"""
    
    # 检查是否有示例数据
    sample_data_path = './dataset/franka_pick_dataset.hdf5'
    if not os.path.exists(sample_data_path):
        print("未找到示例数据文件，正在创建...")
        # 创建示例数据
        from scripts.create_sample_hdf5 import create_sample_hdf5
        create_sample_hdf5(sample_data_path, num_samples=50)
        print("示例数据创建完成")
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 构建训练命令
    cmd = [
        sys.executable,
        'trainer/post_train_vla.py',
        '--data_path', sample_data_path,
        '--epochs', '1',  # 只训练 1 个 epoch 用于测试
        '--batch_size', '2',
        '--learning_rate', '1e-5',
        '--device', device,
        '--hidden_size', '512',
        '--num_hidden_layers', '8',
        '--action_dim', '8',
        '--action_chunk_size', '16',
        '--log_interval', '10',
        '--save_interval', '20',
        '--use_swanlab',  # 启用 SwanLab
        '--swanlab_project', 'MiniMind-VLA-Test',
    ]
    
    print("\n" + "="*80)
    print("开始测试 VLA 训练")
    print("="*80)
    print(f"训练命令: {' '.join(cmd)}")
    print("="*80 + "\n")
    
    # 运行训练
    try:
        result = subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        print("\n" + "="*80)
        print("测试完成！")
        print("="*80)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n训练过程中出现错误: {e}")
        return False
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        return False


if __name__ == "__main__":
    success = test_vla_training()
    sys.exit(0 if success else 1)

