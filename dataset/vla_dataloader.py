"""
VLA 数据集加载器
从 HDF5 文件加载图像、文本和动作数据
"""
import os
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from model.model_vlm import MiniMindVLM
import io

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VLADataset(Dataset):
    """
    VLA 数据集类
    从 HDF5 文件加载图像、文本和动作数据
    """
    def __init__(
        self,
        hdf5_path: str,
        tokenizer,
        preprocess=None,
        max_length: int = 512,
        image_special_token: str = '@' * 196,
        action_dim: int = 8,
        action_chunk_size: int = 100,
        robot_state_dim: int = 8,
    ):
        """
        初始化 VLA 数据集
        
        Args:
            hdf5_path: HDF5 文件路径
            tokenizer: 文本 tokenizer
            preprocess: 图像预处理函数
            max_length: 最大序列长度
            image_special_token: 图像特殊 token
            action_dim: 动作维度
            action_chunk_size: action chunk 大小
            robot_state_dim: 机器人状态维度
        """
        super().__init__()
        self.hdf5_path = hdf5_path
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.max_length = max_length
        self.image_token = image_special_token
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.robot_state_dim = robot_state_dim
        
        # 获取 BOS 和 EOS token IDs
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids
        
        # 打开 HDF5 文件并获取数据长度
        with h5py.File(hdf5_path, 'r') as f:
            # 假设数据存储在 'data' 组中，或者直接是根级别的数据集
            if 'data' in f:
                self.data_group = 'data'
                self.length = len(f['data'])
            elif 'image' in f:
                # 如果直接在根级别，使用第一个数据集的长度
                self.data_group = None
                # 尝试获取长度
                if isinstance(f['image'], h5py.Dataset):
                    self.length = len(f['image'])
                else:
                    # 如果是组，获取第一个子数据集的长度
                    self.length = len(list(f['image'].values())[0])
            else:
                # 默认使用第一个数据集的长度
                first_key = list(f.keys())[0]
                self.data_group = None
                if isinstance(f[first_key], h5py.Dataset):
                    self.length = len(f[first_key])
                else:
                    self.length = 0
        
        # 检查是否包含机器人状态数据
        self.has_robot_state = False
        with h5py.File(hdf5_path, 'r') as f:
            # 正确的检测逻辑：机器人状态存储在每个episode组中
            # 检查是否有任何episode组包含robot_state数据集
            try:
                # 先获取第一个episode组来检查
                if self.data_group:
                    data_grp = f[self.data_group]
                else:
                    data_grp = f
                
                # 检查是否有任何episode组
                for key in data_grp:
                    if key.startswith('episode_'):
                        episode_grp = data_grp[key]
                        if 'robot_state' in episode_grp:
                            self.has_robot_state = True
                            break
            except Exception as e:
                print(f"检查机器人状态时出错: {e}")
        
        print(f"加载 VLA 数据集: {hdf5_path}, 样本数量: {self.length}, 包含机器人状态: {self.has_robot_state}")
    
    def __len__(self):
        return self.length
    
    def _load_image_from_hdf5(self, image_data):
        """
        从 HDF5 数据加载图像
        
        Args:
            image_data: HDF5 数据集或 numpy array
            
        Returns:
            PIL Image 对象
        """
        if isinstance(image_data, h5py.Dataset):
            image_array = image_data[:]
        else:
            image_array = image_data
        
        # 如果是 bytes，需要解码
        if isinstance(image_array, bytes):
            image = Image.open(io.BytesIO(image_array)).convert('RGB')
        elif isinstance(image_array, np.ndarray):
            # 如果是 numpy array，转换为 PIL Image
            if image_array.dtype == np.uint8:
                # 确保形状正确 (H, W, C) 或 (C, H, W)
                if len(image_array.shape) == 3:
                    if image_array.shape[0] == 3 or image_array.shape[0] == 1:
                        # (C, H, W) -> (H, W, C)
                        image_array = np.transpose(image_array, (1, 2, 0))
                    if image_array.shape[2] == 1:
                        # 灰度图转 RGB
                        image_array = np.repeat(image_array, 3, axis=2)
                    image = Image.fromarray(image_array, 'RGB')
                else:
                    raise ValueError(f"不支持的图像形状: {image_array.shape}")
            else:
                # 如果不是 uint8，可能需要归一化
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                image_array = image_array.astype(np.uint8)
                if len(image_array.shape) == 3:
                    if image_array.shape[0] == 3 or image_array.shape[0] == 1:
                        image_array = np.transpose(image_array, (1, 2, 0))
                    if image_array.shape[2] == 1:
                        image_array = np.repeat(image_array, 3, axis=2)
                    image = Image.fromarray(image_array, 'RGB')
                else:
                    raise ValueError(f"不支持的图像形状: {image_array.shape}")
        else:
            raise ValueError(f"不支持的图像数据类型: {type(image_array)}")
        
        return image
    
    def _create_chat_prompt(self, text: str):
        """
        创建聊天提示
        
        Args:
            text: 文本内容
            
        Returns:
            格式化的文本
        """
        # 替换图像占位符
        text = text.replace('<image>', self.image_token)
        
        # 创建消息格式
        messages = [{"role": "user", "content": text}]
        
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except:
            # 如果 apply_chat_template 失败，直接使用原始文本
            prompt = text
        
        return prompt
    
    def _generate_loss_mask(self, input_ids):
        """
        生成损失掩码（只对 assistant 回复部分计算损失）
        
        Args:
            input_ids: token IDs
            
        Returns:
            损失掩码
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
    
    def __getitem__(self, index: int):
        """
        获取单个样本
        
        Returns:
            X: 输入 token IDs [seq_len-1]
            Y: 目标 token IDs [seq_len-1]
            loss_mask: 损失掩码 [seq_len-1]
            pixel_values: 图像张量 [num_images, C, H, W]
            action_targets: 目标动作 [action_chunk_size, action_dim]
            robot_states: 机器人状态 [robot_state_dim]
        """
        robot_state_data = None
        
        with h5py.File(self.hdf5_path, 'r') as f:
            # 正确的加载逻辑：从episode组中加载数据
            try:
                # 获取数据组
                if self.data_group:
                    data_grp = f[self.data_group]
                else:
                    data_grp = f
                
                # 获取所有episode组
                episode_keys = [key for key in data_grp if key.startswith('episode_')]
                
                if episode_keys and index < len(episode_keys):
                    # 按索引获取对应的episode组
                    episode_key = sorted(episode_keys)[index]
                    episode_grp = data_grp[episode_key]
                    
                    # 从episode组中加载数据
                    image_data = episode_grp['rgb'][0]  # 获取第一个图像
                    # 文本数据可能需要从meta或其他地方获取
                    text_data = episode_grp.get('instruction', f"Episode {index}")
                    action_data = episode_grp['action'][0]  # 获取第一个动作
                    
                    # 从episode组中加载机器人状态
                    if 'robot_state' in episode_grp:
                        robot_state_data = episode_grp['robot_state'][0]  # 获取第一个机器人状态
                else:
                    # 回退到旧的加载方式，兼容其他格式
                    if self.data_group:
                        data = f[self.data_group]
                        image_data = data['image'][index] if 'image' in data else f['image'][index]
                        text_data = data['text'][index] if 'text' in data else f['text'][index]
                        action_data = data['action'][index] if 'action' in data else f['action'][index]
                    else:
                        image_data = f['image'][index]
                        text_data = f['text'][index]
                        action_data = f['action'][index]
            except Exception as e:
                print(f"加载数据时出错: {e}")
                # 使用默认值
                image_data = None
                text_data = f"Episode {index}"
                action_data = np.zeros(self.action_chunk_size * self.action_dim, dtype=np.float32)
            
            # 处理文本
            if isinstance(text_data, bytes):
                text = text_data.decode('utf-8')
            elif isinstance(text_data, np.ndarray):
                if text_data.dtype.type is np.str_ or text_data.dtype.type is np.unicode_:
                    text = str(text_data.item())
                else:
                    text = text_data.tobytes().decode('utf-8')
            else:
                text = str(text_data)
            
            # 处理图像
            image = self._load_image_from_hdf5(image_data)
            
            # 处理动作
            if isinstance(action_data, h5py.Dataset):
                action_array = action_data[:]
            else:
                action_array = action_data
            
            if isinstance(action_array, np.ndarray):
                action_array = action_array.astype(np.float32)
            else:
                action_array = np.array(action_array, dtype=np.float32)
            
            # 确保动作形状正确
            if len(action_array.shape) == 1:
                # 如果是 1D，reshape 为 [action_chunk_size, action_dim]
                if len(action_array) == self.action_chunk_size * self.action_dim:
                    action_array = action_array.reshape(self.action_chunk_size, self.action_dim)
                else:
                    # 如果长度不匹配，进行填充或截断
                    target_size = self.action_chunk_size * self.action_dim
                    if len(action_array) < target_size:
                        action_array = np.pad(action_array, (0, target_size - len(action_array)), mode='constant')
                    else:
                        action_array = action_array[:target_size]
                    action_array = action_array.reshape(self.action_chunk_size, self.action_dim)
            elif len(action_array.shape) == 2:
                # 如果已经是 2D，检查形状
                if action_array.shape[0] != self.action_chunk_size or action_array.shape[1] != self.action_dim:
                    # 尝试 reshape
                    if action_array.size == self.action_chunk_size * self.action_dim:
                        action_array = action_array.reshape(self.action_chunk_size, self.action_dim)
                    else:
                        # 截断或填充
                        action_array = action_array[:self.action_chunk_size, :self.action_dim]
                        if action_array.shape[0] < self.action_chunk_size:
                            padding = np.zeros((self.action_chunk_size - action_array.shape[0], self.action_dim), dtype=np.float32)
                            action_array = np.vstack([action_array, padding])
                        if action_array.shape[1] < self.action_dim:
                            padding = np.zeros((self.action_chunk_size, self.action_dim - action_array.shape[1]), dtype=np.float32)
                            action_array = np.hstack([action_array, padding])
        
        # 创建文本提示
        prompt = self._create_chat_prompt(text)
        
        # Tokenize
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self._generate_loss_mask(input_ids)
        
        # 准备输入和目标
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        
        # 处理图像
        if self.preprocess is not None:
            image_tensor = MiniMindVLM.image2tensor(image, self.preprocess)
        else:
            # 如果没有 preprocess，使用简单的转换
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            image_tensor = transform(image)
        
        # 图像形状: [C, H, W]，DataLoader 会自动添加 batch 维度
        # 为了与 VLM 兼容，需要添加 num_images 维度: [1, C, H, W]
        # 这样在 batch 后变成 [batch, 1, C, H, W]，符合模型期望
        image_tensors = image_tensor.unsqueeze(0)  # [1, C, H, W]
        
        # 转换为动作张量
        action_targets = torch.tensor(action_array, dtype=torch.float32)
        
        # 处理机器人状态数据
        if robot_state_data is not None:
            if isinstance(robot_state_data, h5py.Dataset):
                robot_state_array = robot_state_data[:]
            else:
                robot_state_array = robot_state_data
            
            if isinstance(robot_state_array, np.ndarray):
                robot_state_array = robot_state_array.astype(np.float32)
            else:
                robot_state_array = np.array(robot_state_array, dtype=np.float32)
            
            # 确保机器人状态形状正确
            if len(robot_state_array.shape) == 0:
                # 标量，转换为单元素数组
                robot_state_array = np.array([robot_state_array], dtype=np.float32)
            elif len(robot_state_array.shape) > 1:
                # 多维数组，展平
                robot_state_array = robot_state_array.flatten()
            
            # 截断或填充到指定维度
            if len(robot_state_array) < self.robot_state_dim:
                robot_state_array = np.pad(robot_state_array, (0, self.robot_state_dim - len(robot_state_array)), mode='constant')
            else:
                robot_state_array = robot_state_array[:self.robot_state_dim]
        else:
            # 如果没有机器人状态数据，创建零张量
            robot_state_array = np.zeros(self.robot_state_dim, dtype=np.float32)
        
        # 转换为张量
        robot_states = torch.tensor(robot_state_array, dtype=torch.float32)
        
        return X, Y, loss_mask, image_tensors, action_targets, robot_states

