import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# 设置随机种子以保证可复现性
torch.manual_seed(0)
np.random.seed(0)

# ------------------------------------------------------
# 1. 实验配置（遵循原文设定）
# ------------------------------------------------------
class Config:
    d = 2               # 底层数据维度（原文d=2）
    D_list = [2, 8, 16, 512]  # 观测空间维度（原文测试的D值）
    hidden_dim = 256    # MLP隐藏层维度（原文5-layer ReLU MLP）
    num_layers = 5      # MLP层数（实际为6层：5个隐藏层+1个输出层）
    epochs = 10000       # 训练轮次
    batch_size = 256    # 批次大小
    lr = 1e-3           # 学习率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = 1000 # 训练数据总量（增加以提升效果）
    noise_scale = 1.0   # 噪声幅度
    ode_steps = 250     # ODE 积分步数（用于生成过程）
    data_type = "spiral" # 数据生成类型："moons" 或 "spiral"

config = Config()

# ------------------------------------------------------
# 2. 底层数据生成函数
# ------------------------------------------------------
def sample_spiral_2d(n_points):
    """生成螺旋形2D数据（类似 train_toy_exp.py）"""
    theta = np.linspace(0, 4 * np.pi, n_points)
    r = theta / (4 * np.pi) * 2.0
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    pts = np.stack([x, y], axis=1)
    pts += 0.02 * np.random.randn(*pts.shape)
    return torch.from_numpy(pts).float()

def sample_moons_2d(n_points):
    """生成双月形2D数据"""
    from sklearn.datasets import make_moons
    x_underlying, _ = make_moons(n_samples=n_points, noise=0.05, random_state=42)
    return torch.tensor(x_underlying, dtype=torch.float32)

# ------------------------------------------------------
# 3. 生成底层数据与高维投影（原文核心设定）
# ------------------------------------------------------
class HighDimDataset(Dataset):
    def __init__(self, d, D, num_samples, data_type="moons", device=None):
        super().__init__()
        self.d = d
        self.D = D
        self.num_samples = num_samples
        self.device = device if device is not None else config.device
        
        # 步骤1：生成2维底层数据
        if data_type == "spiral":
            self.x_underlying = sample_spiral_2d(num_samples)
        else:  # "moons"
            self.x_underlying = sample_moons_2d(num_samples)
        
        # 步骤2：生成随机列正交投影矩阵P（D×d），将底层数据投影到D维观测空间
        # 用QR分解保证列正交：P = Q（正交矩阵）的前d列
        P = torch.randn(D, d, dtype=torch.float32)  # 在CPU上创建
        Q, _ = torch.linalg.qr(P)  # Q是D×d的列正交矩阵（P^T P = I_d）
        self.P = Q  # 投影矩阵（模型未知，仅用于可视化）
        
        # 步骤3：生成高维干净数据x = x_underlying @ P.T（N, D）
        self.x_clean = torch.matmul(self.x_underlying, self.P.T)  # (N, d) @ (d, D) = (N, D)
        
        # 步骤4：计算数据归一化参数（用于训练）
        self.sigma = self.x_clean.std().item()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.x_clean[idx]  # 返回高维干净数据

# ------------------------------------------------------
# 4. 生成模型（改进的MLP架构，类似 train_toy_exp.py）
# ------------------------------------------------------
class MLPGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        layers = []
        # 构建 MLP：输入层 + num_layers 个隐藏层 + 输出层
        # 实际总层数为 num_layers + 1（隐藏层数 + 输出层）
        dims = [input_dim + 1] + [hidden_dim] * num_layers + [output_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))  # 使用 inplace 以节省内存
        # 输出层（不使用激活函数）
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
    
    def forward(self, z_t, t):
        """
        z_t: (B, D) 含噪数据
        t: (B,) 或 (B, 1) 时间步（归一化到[0,1]）
        """
        # 处理时间步维度
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        # 拼接输入：D维含噪数据 + 1维时间步
        inp = torch.cat([z_t, t], dim=-1)  # (B, D+1)
        return self.net(inp)  # (B, D)

# ------------------------------------------------------
# 5. 训练函数（支持x/ϵ/v三种预测目标，统一用v-loss计算，改进版）
# ------------------------------------------------------
def train_prediction_target(D, target_type):
    """
    训练单个预测目标的模型
    D: 观测空间维度
    target_type: 预测目标类型（'x'→x-prediction, 'eps'→ϵ-prediction, 'v'→v-prediction）
    """
    # 1. 初始化数据集与数据加载器
    dataset = HighDimDataset(d=config.d, D=D, num_samples=config.num_samples, 
                            data_type=config.data_type, device=config.device)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    sigma = dataset.sigma  # 数据归一化参数
    
    # 2. 初始化模型、损失函数、优化器
    model = MLPGenerator(
        input_dim=D, 
        output_dim=D,  # 输出维度=观测空间维度D
        hidden_dim=config.hidden_dim, 
        num_layers=config.num_layers
    ).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # 3. 训练循环（统一使用 v-loss，改进的噪声混合公式）
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0.0
        num_batches = 0
        for x_clean in dataloader:
            x_clean = x_clean.to(config.device)  # (B, D)
            B = x_clean.shape[0]
            
            # 步骤1：数据归一化（参考 train_toy_exp.py）
            x_clean_normalized = x_clean / sigma
            
            # 步骤2：采样时间步t（均匀分布[0, 1]，使用 clamp 避免除以0）
            t = torch.rand(B, device=config.device).unsqueeze(-1)  # (B, 1)
            
            # 步骤3：生成噪声（标准高斯噪声）
            noise = torch.randn_like(x_clean_normalized) * config.noise_scale  # (B, D)
            
            # 步骤4：计算含噪数据z_t（统一公式：z_t = (1-t)*x + t*noise）
            z_t = (1 - t) * x_clean_normalized + t * noise  # (B, D)
            
            # 步骤5：模型预测
            pred = model(z_t, t.squeeze(-1))  # (B, D)，时间步可以是 (B,) 或 (B,1)
            
            # 步骤6：根据预测目标类型计算损失（统一使用 v-loss）
            if target_type == 'x':
                # x-prediction：模型输出 x_hat，转换为 v_pred
                dnorm = torch.clamp(t, min=0.05)  # 避免除以0
                v_true = (x_clean_normalized - z_t) / dnorm
                v_pred = (pred - z_t) / dnorm
                loss = ((v_true - v_pred) ** 2).mean()
            elif target_type == 'eps':
                # ϵ-prediction：模型输出 eps_hat，需要转换为 v_pred
                # 从 eps_hat 反推 x_hat: z_t = (1-t)*x_hat + t*eps_hat
                # 因此: x_hat = (z_t - t*eps_hat) / (1-t)
                dnorm_t = torch.clamp(t, min=0.05)
                dnorm_1t = torch.clamp(1 - t, min=0.05)
                eps_hat = pred
                x_hat = (z_t - t * eps_hat) / dnorm_1t
                v_true = (x_clean_normalized - z_t) / dnorm_t
                v_pred = (x_hat - z_t) / dnorm_t
                loss = ((v_true - v_pred) ** 2).mean()
            elif target_type == 'v':
                # v-prediction：模型直接输出 v_pred
                dnorm = torch.clamp(t, min=0.05)
                v_true = (x_clean_normalized - z_t) / dnorm
                v_pred = pred
                loss = ((v_true - v_pred) ** 2).mean()
            else:
                raise ValueError("target_type must be 'x', 'eps', or 'v'")
            
            # 步骤7：反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * B
            num_batches += 1
        
        # 打印训练进度
        avg_loss = total_loss / (num_batches * config.batch_size)
        if (epoch + 1) % 200 == 0 or (epoch + 1) == config.epochs:
            print(f"[D={D}, Target={target_type}] Epoch {epoch+1}/{config.epochs}, Avg Loss: {avg_loss:.6f}")
    
    # 训练完成后返回模型、数据集和归一化参数
    return model, dataset, sigma

# ------------------------------------------------------
# 6. 生成与可视化函数（改进的多步 ODE 积分生成）
# ------------------------------------------------------
def generate_and_visualize(model, dataset, D, target_type, num_gen=2000, sigma=1.0, ode_steps=250):
    """
    生成高维数据并投影回2维可视化
    使用多步 ODE 积分提高生成质量（参考 train_toy_exp.py）
    """
    model.eval()
    P = dataset.P.to(config.device)  # 确保投影矩阵在正确的设备上
    
    with torch.no_grad():
        # 1. 从标准高斯噪声开始生成
        x = torch.randn(num_gen, D, dtype=torch.float32, device=config.device)
        dt = 1.0 / ode_steps  # 时间步长
        
        # 2. 多步 ODE 积分（从 t=1 到 t=0）
        for i in range(ode_steps, 0, -1):
            t = torch.full((num_gen,), i * dt, device=config.device)  # (num_gen,)
            x_t = x  # 当前状态
            
            # 模型预测
            pred = model(x_t, t)  # (num_gen, D)
            
            # 根据预测目标类型计算流速 v_pred
            t_expanded = t.unsqueeze(-1)  # (num_gen, 1)
            if target_type == 'x':
                # x-prediction：模型输出 x_hat，计算 v_pred = (x_t - x_hat) / t
                # 参考 train_toy_exp.py: vp = (x_t - pred) / t[:,None]
                dnorm = torch.clamp(t, min=0.05).unsqueeze(-1)
                v_pred = (x_t - pred) / dnorm  # (num_gen, D)
            elif target_type == 'eps':
                # ϵ-prediction：模型输出 eps_hat，需要转换为 v_pred
                # 从 eps_hat 反推 x_hat: x_t = (1-t)*x_hat + t*eps_hat
                # 因此: x_hat = (x_t - t*eps_hat) / (1-t)
                dnorm_t = torch.clamp(t, min=0.05).unsqueeze(-1)
                dnorm_1t = torch.clamp(1 - t, min=0.05).unsqueeze(-1)
                eps_hat = pred
                x_hat = (x_t - t_expanded * eps_hat) / dnorm_1t
                v_pred = (x_hat - x_t) / dnorm_t
            elif target_type == 'v':
                # v-prediction：模型直接输出 v_pred
                v_pred = pred  # (num_gen, D)
            else:
                raise ValueError("target_type must be 'x', 'eps', or 'v'")
            
            # ODE 更新：dx/dt = -v，因此 x_new = x_old - dt * v
            x = x_t - dt * v_pred
        
        # 3. 反归一化生成的数据
        x_gen = x * sigma
        
        # 4. 用数据集的投影矩阵P将x_gen从D维投影回2维（用于可视化）
        x_gen_2d = torch.matmul(x_gen, P).cpu().numpy()  # (num_gen, D) @ (D, d) = (num_gen, d)
        # 底层真实数据的2维分布（用于对比）
        x_underlying_2d = dataset.x_underlying.numpy()
    
    # 5. 绘制可视化结果
    plt.figure(figsize=(8, 4))
    # 子图1：底层真实数据分布
    plt.subplot(1, 2, 1)
    plt.scatter(x_underlying_2d[:, 0], x_underlying_2d[:, 1], s=5, alpha=0.3, c='blue', label='True Underlying Data (d=2)')
    # plt.title(f'True 2D Underlying Data')
    # plt.legend()
    plt.axis('equal')
    # 子图2：生成数据投影后的分布
    plt.subplot(1, 2, 2)
    plt.scatter(x_gen_2d[:, 0], x_gen_2d[:, 1], s=5, alpha=0.7, c='red', label=f'Generated Data (D={D}, {target_type}-pred)')
    # plt.title(f'Generated Data Projected to 2D (D={D}, {target_type}-pred)')
    # plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{config.data_type}_figure2_toy_experiment_D{D}_{target_type}pred.png', dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------
# 7. 主函数：运行所有实验并可视化
# ------------------------------------------------------
def main():
    # 遍历所有观测维度D和预测目标
    target_types = ['x', 'eps', 'v']
    for D in config.D_list:
        print(f"\n=== Starting Experiment: D={D} ===")
        for target_type in target_types:
            print(f"\n--- Training {target_type}-prediction for D={D} ---")
            # 训练模型
            model, dataset, sigma = train_prediction_target(D=D, target_type=target_type)
            # 生成并可视化结果（使用多步 ODE 积分）
            generate_and_visualize(model, dataset, D=D, target_type=target_type, 
                                  num_gen=2000, sigma=sigma, ode_steps=config.ode_steps)
    print("\nAll experiments completed! Results saved as 'figure2_toy_experiment_*.png'")

if __name__ == "__main__":
    main()