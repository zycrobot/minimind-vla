"""
创建示例 HDF5 数据文件用于测试 VLA 模型
"""
import h5py
import numpy as np
from PIL import Image
import os

def create_sample_hdf5(output_path='./dataset/sample_vla_data.hdf5', num_samples=100):
    """
    创建示例 HDF5 数据文件
    
    Args:
        output_path: 输出文件路径
        num_samples: 样本数量
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # 创建示例数据
    action_dim = 7
    action_chunk_size = 16
    
    with h5py.File(output_path, 'w') as f:
        # 创建数据集
        # 图像数据: 存储为 numpy array (224, 224, 3)
        image_shape = (num_samples, 224, 224, 3)
        image_dset = f.create_dataset('image', shape=image_shape, dtype=np.uint8)
        
        # 文本数据: 存储为字符串
        text_dset = f.create_dataset('text', shape=(num_samples,), dtype=h5py.string_dtype(encoding='utf-8'))
        
        # 动作数据: 存储为 numpy array (action_chunk_size, action_dim)
        action_shape = (num_samples, action_chunk_size, action_dim)
        action_dset = f.create_dataset('action', shape=action_shape, dtype=np.float32)
        
        # 填充数据
        for i in range(num_samples):
            # 创建随机图像
            image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            image_array[:,:,:]=i
            image_dset[i] = image_array
            
            # 创建示例文本
            text = f"请执行抓取动作，目标位置为 ({i % 10}, {i % 10 + 1})"
            text_dset[i] = text
            
            # 创建示例动作（随机动作序列）
            # 动作范围通常在 [-1, 1] 或 [0, 1] 之间
            action_array = np.random.uniform(-1.0, 1.0, (action_chunk_size, action_dim)).astype(np.float32)
            action_array[:,:] = i
            action_dset[i] = action_array
    
    print(f"成功创建示例 HDF5 文件: {output_path}")
    print(f"  样本数量: {num_samples}")
    print(f"  图像形状: (224, 224, 3)")
    print(f"  动作形状: ({action_chunk_size}, {action_dim})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="创建示例 HDF5 数据文件")
    parser.add_argument('--output_path', type=str, default='./dataset/sample_vla_data.hdf5', 
                       help='输出文件路径')
    parser.add_argument('--num_samples', type=int, default=100, 
                       help='样本数量')
    args = parser.parse_args()
    
    create_sample_hdf5(args.output_path, args.num_samples)

