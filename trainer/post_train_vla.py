import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from model.model_vla import MiniMindVLA, VLAConfig
from dataset.vla_dataloader import VLADataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, SkipBatchSampler

warnings.filterwarnings('ignore')


def init_vla_model(vla_config, from_weight='pretrain_vlm', tokenizer_path='model', 
                   vision_model_path='./model/vision_model/clip-vit-base-patch16', 
                   save_dir='out', device='cuda', freeze_vision=True):
    """
    初始化 VLA 模型
    
    Args:
        vla_config: VLA 配置
        from_weight: 预训练权重名称
        tokenizer_path: tokenizer 路径
        vision_model_path: 视觉模型路径
        save_dir: 保存目录
        device: 设备
        freeze_vision: 是否冻结视觉编码器
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 如果本地路径不存在，尝试使用 HuggingFace 模型名称
    if not os.path.exists(vision_model_path):
        Logger(f'警告: 视觉模型路径不存在: {vision_model_path}')
        Logger(f'将尝试从 HuggingFace 下载模型...')
        # 使用 HuggingFace 模型名称
        vision_model_path = "openai/clip-vit-base-patch16"
    
    model = MiniMindVLA(vla_config, vision_model_path=vision_model_path)
    
    # 检查视觉编码器是否成功加载
    if model.vision_encoder is None:
        Logger(f'错误: 视觉编码器加载失败！')
        Logger(f'请检查视觉模型路径: {vision_model_path}')
        Logger(f'或者确保网络连接正常，可以从 HuggingFace 下载模型')
        raise ValueError(f"视觉编码器加载失败，路径: {vision_model_path}")
    
    # 加载预训练权重
    if from_weight != 'none':
        moe_suffix = '_moe' if vla_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{vla_config.hidden_size}{moe_suffix}.pth'
        if os.path.exists(weight_path):
            Logger(f'加载预训练权重: {weight_path}')
            weights = torch.load(weight_path, map_location=device)
            model.load_state_dict(weights, strict=False)
        else:
            Logger(f'警告: 预训练权重不存在: {weight_path}，使用随机初始化')
    
    # 固定视觉编码器
    if freeze_vision:
        for name, param in model.named_parameters():
            if 'vision_encoder' in name:
                param.requires_grad = False
                # Logger(f'冻结参数: {name}')
    
    # 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f'VLA 模型总参数量: {total_params / 1e6:.3f} 百万')
    Logger(f'VLA 模型可训练参数量: {trainable_params / 1e6:.3f} 百万')
    
    preprocess = model.processor
    return model.to(device), tokenizer, preprocess


def vla_checkpoint(vla_config, weight='post_train_vla', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='checkpoints', **kwargs):
    """
    保存或加载 VLA 模型检查点
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if vla_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{vla_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{vla_config.hidden_size}{moe_path}_resume.pth'
    
    if model is not None:
        from torch.nn.parallel import DistributedDataParallel
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        # 移除 vision_encoder 参数（不需要保存，因为是预训练的）
        clean_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vision_encoder.')}
        ckp_tmp = ckp_path + '.tmp'
        torch.save({k: v.half() for k, v in clean_state_dict.items()}, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)
        
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value
        
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """训练一个 epoch"""
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    
    for step, (X, Y, loss_mask, pixel_values, action_targets, robot_states) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        pixel_values = pixel_values.to(args.device)
        action_targets = action_targets.to(args.device)
        robot_states = robot_states.to(args.device) if robot_states is not None else None
        
        # 学习率调度
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # 前向传播
            try:
                res = model(
                    input_ids=X,
                    pixel_values=pixel_values,
                    robot_states=robot_states,
                    action_targets=action_targets,
                    return_actions=True,
                )
            except Exception as e:
                Logger(f"前向传播错误: {e}")
                Logger(f"X shape: {X.shape}")
                Logger(f"pixel_values shape: {pixel_values.shape if pixel_values is not None else None}")
                Logger(f"action_targets shape: {action_targets.shape if action_targets is not None else None}")
                raise
            
            # 计算语言模型损失
            lm_loss_tensor = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            mask_sum = loss_mask.sum()
            if mask_sum.item() == 0:
                lm_loss = torch.zeros(1, device=args.device, dtype=lm_loss_tensor.dtype)
            else:
                lm_loss = (lm_loss_tensor * loss_mask).sum() / mask_sum
            
            # 计算 action 损失
            action_loss = res.get('action_loss', torch.tensor(0.0, device=args.device))
            
            # 总损失
            total_loss = lm_loss + args.action_loss_weight * action_loss
            total_loss += res.aux_loss if hasattr(res, 'aux_loss') and res.aux_loss is not None else 0.0
            total_loss = total_loss / args.accumulation_steps

        scaler.scale(total_loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # 每个step都记录loss
        spend_time = time.time() - start_time
        current_lm_loss = lm_loss.item() * args.accumulation_steps
        current_action_loss = action_loss.item() * args.accumulation_steps if isinstance(action_loss, torch.Tensor) else action_loss * args.accumulation_steps
        current_total_loss = total_loss.item() * args.accumulation_steps
        current_lr = optimizer.param_groups[-1]['lr']
        eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
        
        Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) '  
               f'total_loss:{current_total_loss:.6f} '  
               f'lm_loss:{current_lm_loss:.6f} '  
               f'action_loss:{current_action_loss:.6f} '  
               f'lr:{current_lr:.12f} '  
               f'epoch_Time:{eta_min}min')
        
        if wandb:
            wandb.log({
                "total_loss": current_total_loss,
                "lm_loss": current_lm_loss,
                "action_loss": current_action_loss,
                "lr": current_lr,
                "epoch_Time": eta_min,
                "step": epoch * iters + step
            })

        # 只在指定间隔保存模型，避免频繁保存
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if vla_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{vla_config.hidden_size}{moe_suffix}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            clean_state_dict = {
                key: value for key, value in state_dict.items() if not key.startswith('vision_encoder.')
            }
            clean_state_dict = {k: v.half() for k, v in clean_state_dict.items()}  # 半精度保存
            torch.save(clean_state_dict, ckp)
            vla_checkpoint(vla_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='checkpoints', scaler=scaler)
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-VLA Post Training")
    
    # 基本参数
    parser.add_argument("--save_dir", type=str, default="out", help="模型保存目录")
    parser.add_argument('--save_weight', default='post_train_vla', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=50, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="模型保存间隔")
    
    # 模型参数
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1536, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    
    # VLA 特定参数
    parser.add_argument('--action_dim', default=8, type=int, help="动作维度")
    parser.add_argument('--action_chunk_size', default=100, type=int, help="action chunk 大小")
    parser.add_argument('--action_hidden_size', default=256, type=int, help="action 模块隐藏层大小")
    parser.add_argument('--action_loss_weight', default=1.0, type=float, help="action 损失权重")
    parser.add_argument('--robot_state_dim', default=8, type=int, help="机器人状态维度，默认7个关节角度+1个 gripper 状态")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, default="./dataset/vla_data.hdf5", help="训练数据路径（HDF5文件）")
    parser.add_argument('--from_weight', default='pretrain_vlm', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    
    # SwanLab 参数
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用 SwanLab")
    parser.add_argument("--swanlab_project", type=str, default="MiniMind-VLA", help="SwanLab 项目名")
    
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    vla_config = VLAConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        use_moe=bool(args.use_moe),
        action_dim=args.action_dim,
        action_chunk_size=args.action_chunk_size,
        action_hidden_size=args.action_hidden_size,
        robot_state_dim=args.robot_state_dim,
    )
    ckp_data = vla_checkpoint(vla_config, weight=args.save_weight, save_dir='checkpoints') if args.from_resume == 1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置 SwanLab ==========
    wandb = None
    if args.use_swanlab and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-VLA-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(
            project=args.swanlab_project,
            name=wandb_run_name,
            id=wandb_id,
            resume=resume,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "hidden_size": args.hidden_size,
                "num_hidden_layers": args.num_hidden_layers,
                "action_dim": args.action_dim,
                "action_chunk_size": args.action_chunk_size,
                "action_loss_weight": args.action_loss_weight,
                "robot_state_dim": args.robot_state_dim,
            }
        )
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer, preprocess = init_vla_model(
        vla_config,
        from_weight=args.from_weight,
        device=args.device,
        freeze_vision=True  # 固定视觉编码器
    )
    
    train_ds = VLADataset(
        hdf5_path=args.data_path,
        tokenizer=tokenizer,
        preprocess=preprocess,
        max_length=vla_config.max_seq_len,
        image_special_token=vla_config.image_special_token,
        action_dim=args.action_dim,
        action_chunk_size=args.action_chunk_size,
        robot_state_dim=args.robot_state_dim,
    )
    
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
        else:  # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), 0, wandb)

