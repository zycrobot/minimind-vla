import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from .model_vlm import MiniMindVLM, VLMConfig, VisionProj
from .model_minimind import MOEFeedForward
from typing import Optional, Tuple, List, Union
from transformers import CLIPProcessor, CLIPModel

warnings.filterwarnings('ignore')


class VLAConfig(VLMConfig):
    model_type = "minimind-vla"

    def __init__(
            self,
            action_dim: int = 8,  # 动作维度（例如：7DOF机械臂 + gripper）
            action_chunk_size: int = 100,  # action chunk 的大小
            flow_matching_num_steps: int = 100,  # flow matching 的步数
            flow_matching_sigma: float = 0.1,  # flow matching 的噪声标准差
            action_hidden_size: int = 256,  # action 模块的隐藏层大小
            action_prediction_target: str = "x",  # 预测模式：x / eps / v
            robot_state_dim: int = 8,  # 机器人状态维度（例如：7DOF关节角度 + gripper）
            **kwargs,
    ):
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.flow_matching_num_steps = flow_matching_num_steps
        self.flow_matching_sigma = flow_matching_sigma
        self.action_hidden_size = action_hidden_size
        self.action_prediction_target = action_prediction_target
        self.robot_state_dim = robot_state_dim
        super().__init__(**kwargs)


class FlowMatchingAction(nn.Module):
    """
    基于 Flow Matching 方法的 Action 预测模块
    用于预测机器人动作序列（action chunk）
    """
    def __init__(
            self,
            hidden_size: int,
            action_dim: int,
            action_chunk_size: int,
            action_hidden_size: int = 256,
            num_layers: int = 3,
            target_type: str = "x",
            flow_matching_sigma: float = 0.1,
            num_sampling_steps: int = 100,
            robot_state_dim: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.action_hidden_size = action_hidden_size
        self.robot_state_dim = robot_state_dim
        if target_type not in ("x", "eps", "v"):
            raise ValueError("action target_type 必须是 'x', 'eps' 或 'v'")
        self.target_type = target_type
        self.flow_matching_sigma = flow_matching_sigma
        self.num_sampling_steps = num_sampling_steps
        
        # 将视觉-语言特征投影到 action 空间
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_size, action_hidden_size),
            nn.LayerNorm(action_hidden_size),
            nn.GELU(),
            nn.Linear(action_hidden_size, action_hidden_size),
        )
        
        # 机器人状态特征投影
        self.robot_state_proj = nn.Sequential(
            nn.Linear(robot_state_dim, action_hidden_size),
            nn.LayerNorm(action_hidden_size),
            nn.GELU(),
            nn.Linear(action_hidden_size, action_hidden_size),
        )
        
        # Flow Matching 网络：预测速度场 v_t(x_t, t, context, robot_state)
        # 输入：当前状态 x_t, 时间 t, 上下文特征 context, 机器人状态 robot_state
        # 输出：速度场 v_t
        layers = []
        input_dim = action_dim * action_chunk_size + action_hidden_size + 1  # x_t + context(包含robot_state) + t
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else action_hidden_size, action_hidden_size))
            layers.append(nn.LayerNorm(action_hidden_size))
            layers.append(nn.GELU())
        
        layers.append(nn.Linear(action_hidden_size, action_dim * action_chunk_size))
        self.velocity_net = nn.Sequential(*layers)
        
    def forward(
            self,
            hidden_states: torch.Tensor,
            robot_states: Optional[torch.Tensor] = None,
            action_targets: Optional[torch.Tensor] = None,
            training: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            hidden_states: 视觉-语言融合特征 [batch, seq_len, hidden_size]
            robot_states: 机器人状态 [batch, robot_state_dim]
            action_targets: 目标动作序列 [batch, action_chunk_size, action_dim] (训练时使用)
            training: 是否为训练模式
            
        Returns:
            如果 training=True: (predicted_actions, loss)
            如果 training=False: predicted_actions
        """
        batch_size = hidden_states.size(0)
        
        # 提取上下文特征（使用最后一个 token 的特征）
        context = hidden_states[:, -1, :]  # [batch, hidden_size]
        context = self.feature_proj(context)  # [batch, action_hidden_size]
        
        # 处理机器人状态
        if robot_states is None:
            # 如果没有提供机器人状态，创建零张量作为默认值
            robot_states = torch.zeros(batch_size, self.robot_state_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # 投影机器人状态特征
        robot_state_feat = self.robot_state_proj(robot_states)  # [batch, action_hidden_size]
        
        # 合并上下文特征和机器人状态特征
        combined_context = context + robot_state_feat
        
        if training and action_targets is not None:
            # 训练模式：使用 flow matching 损失
            loss = self.compute_flow_matching_loss(combined_context, action_targets)
            # 推理时预测动作（用于辅助观测）
            predicted_actions = self.sample_actions(combined_context)
            return predicted_actions, loss
        else:
            # 推理模式：采样动作
            predicted_actions = self.sample_actions(combined_context)
            return predicted_actions
    
    def compute_flow_matching_loss(
            self,
            context: torch.Tensor,
            action_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 Flow Matching 损失
        
        Args:
            context: 上下文特征 [batch, action_hidden_size]
            action_targets: 目标动作序列 [batch, action_chunk_size, action_dim]
            
        Returns:
            flow matching 损失
        """
        batch_size = action_targets.size(0)
        action_targets_flat = action_targets.view(batch_size, -1)  # [batch, action_chunk_size * action_dim]
        
        # 随机采样时间步 t ~ U(0, 1)
        t = torch.rand(batch_size, 1, device=action_targets.device)
        t_clamped = torch.clamp(t, min=0.05)
        
        # 从噪声分布采样 x_0 ~ N(0, sigma^2)
        noise = torch.randn_like(action_targets_flat) * self.flow_matching_sigma
        
        # 插值路径：z_t = (1 - t) * x_clean + t * noise
        z_t = (1 - t) * action_targets_flat + t * noise
        
        # 真实速度场（v_true）
        v_true = (action_targets_flat - z_t) / t_clamped
        
        # 预测
        v_pred = self._predict_velocity(z_t, context, t)
        
        # 计算 MSE 损失
        loss = F.mse_loss(v_pred, v_true)
        
        return loss

    def _predict_velocity(
            self,
            x_t: torch.Tensor,
            context: torch.Tensor,
            t: torch.Tensor,
    ) -> torch.Tensor:
        """
        根据 target_type 将网络输出转换为速度场
        """
        # 注意：context 已经包含了机器人状态信息
        net_input = torch.cat([x_t, context, t], dim=1)
        pred = self.velocity_net(net_input)
        return self._convert_prediction_to_velocity(pred, x_t, t)

    def _convert_prediction_to_velocity(
            self,
            pred: torch.Tensor,
            x_t: torch.Tensor,
            t: torch.Tensor,
    ) -> torch.Tensor:
        t_clamped = torch.clamp(t, min=0.05)
        if self.target_type == "x":
            return (pred - x_t) / t_clamped
        if self.target_type == "eps":
            one_minus_t = torch.clamp(1 - t, min=0.05)
            x_hat = (x_t - t * pred) / one_minus_t
            return (x_hat - x_t) / t_clamped
        # self.target_type == "v"
        return pred
    
    def sample_actions(
            self,
            context: torch.Tensor,
            num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        使用 Flow Matching 采样动作序列
        
        Args:
            context: 上下文特征 [batch, action_hidden_size]
            num_steps: ODE 求解的步数
            
        Returns:
            预测的动作序列 [batch, action_chunk_size, action_dim]
        """
        num_steps = num_steps or self.num_sampling_steps
        batch_size = context.size(0)
        device = context.device
        dtype = context.dtype
        
        # 从噪声分布初始化：x_1 ~ N(0, sigma^2)
        x = torch.randn(batch_size, self.action_chunk_size * self.action_dim, device=device, dtype=dtype) * self.flow_matching_sigma
        
        # 使用 Euler 方法从 t=1 积分到 t=0：dx/dt = -v_t(x_t, t)
        dt = 1.0 / num_steps
        for i in range(num_steps, 0, -1):
            t = torch.full((batch_size, 1), i * dt, device=device, dtype=dtype)
            v = self._predict_velocity(x, context, t)
            x = x - dt * v
        
        # 重塑为动作序列
        actions = x.view(batch_size, self.action_chunk_size, self.action_dim)
        
        return actions


# 继承自 VLM 模型
class MiniMindVLA(MiniMindVLM):
    config_class = VLAConfig

    def __init__(
            self,
            params: VLAConfig = None,
            vision_model_path: str = "./model/vision_model/clip-vit-base-patch16"
    ):
        super().__init__(params, vision_model_path)
        if not params:
            params = VLAConfig()
        self.params = params
        
        # 添加 action 预测模块
        self.action_head = FlowMatchingAction(
            hidden_size=params.hidden_size,
            action_dim=params.action_dim,
            action_chunk_size=params.action_chunk_size,
            action_hidden_size=params.action_hidden_size,
            target_type=params.action_prediction_target,
            flow_matching_sigma=params.flow_matching_sigma,
            num_sampling_steps=params.flow_matching_num_steps,
            robot_state_dim=params.robot_state_dim,
        )

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            pixel_values: Optional[torch.FloatTensor] = None,
            robot_states: Optional[torch.Tensor] = None,
            action_targets: Optional[torch.Tensor] = None,
            return_actions: bool = True,
            **args
    ):
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            past_key_values: 缓存的键值对
            use_cache: 是否使用缓存
            logits_to_keep: 保留的 logits 数量
            pixel_values: 图像像素值
            action_targets: 目标动作序列 [batch, action_chunk_size, action_dim] (训练时使用)
            return_actions: 是否返回预测的动作
            **args: 其他参数
            
        Returns:
            包含 logits、hidden_states、actions 等的输出字典
        """
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        # 安全地获取 start_pos
        if past_key_values[0] is not None and isinstance(past_key_values[0], tuple) and len(past_key_values[0]) > 0:
            start_pos = past_key_values[0][0].shape[1]
        else:
            start_pos = 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        # 处理视觉输入
        if pixel_values is not None and start_pos == 0:
            # 处理不同的输入形状
            if len(pixel_values.shape) == 6:
                pixel_values = pixel_values.squeeze(2)
            elif len(pixel_values.shape) == 4:
                # 如果是 [batch, C, H, W]，添加 num_images 维度
                pixel_values = pixel_values.unsqueeze(1)  # [batch, 1, C, H, W]
            elif len(pixel_values.shape) == 5:
                # 已经是 [batch, num_images, C, H, W]，保持不变
                pass
            else:
                raise ValueError(f"不支持的 pixel_values 形状: {pixel_values.shape}")
            
            bs, num, c, im_h, im_w = pixel_values.shape
            stack_dim = 1 if bs > 1 else 0
            vision_tensors = torch.stack([
                MiniMindVLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)
                for i in range(num)
            ], dim=stack_dim)
            hidden_states = self.count_vision_proj(
                tokens=input_ids,
                h=hidden_states,
                vision_tensors=vision_tensors,
                seqlen=input_ids.shape[1]
            )

        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        # 计算语言模型损失
        aux_loss = sum(
            getattr(layer.mlp, 'aux_loss', 0.0)
            for layer in self.model.layers
            if isinstance(layer.mlp, MOEFeedForward) and hasattr(layer.mlp, 'aux_loss')
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        # 预测动作
        action_loss = None
        predicted_actions = None
        if return_actions:
            training = self.training and action_targets is not None
            action_output = self.action_head(
                hidden_states=hidden_states,
                robot_states=robot_states,
                action_targets=action_targets,
                training=training
            )
            
            if training:
                predicted_actions, action_loss = action_output
            else:
                predicted_actions = action_output

        # 构建输出
        self.OUT.__setitem__('last_hidden_state', hidden_states)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', presents)
        
        if return_actions and predicted_actions is not None:
            self.OUT.__setitem__('actions', predicted_actions)
        if action_loss is not None:
            self.OUT.__setitem__('action_loss', action_loss)
        
        return self.OUT

    def predict_actions(
            self,
            input_ids: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            robot_states: Optional[torch.Tensor] = None,
            **kwargs
    ) -> torch.Tensor:
        """
        便捷方法：仅预测动作
        
        Args:
            input_ids: 输入 token IDs
            pixel_values: 图像像素值
            robot_states: 机器人状态 [batch, robot_state_dim]
            **kwargs: 其他参数
            
        Returns:
            预测的动作序列 [batch, action_chunk_size, action_dim]
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                robot_states=robot_states,
                return_actions=True,
                **kwargs
            )
            return output.get('actions', None)

