import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import permutations

# ==========================================
# 1. Conditional VAE
# ==========================================
class ConditionalVAE(nn.Module):
    def __init__(self, K, latent_dim=16):
        super().__init__()
        self.K = K
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),  # <--- 新增：防止过拟合
            nn.Linear(784, 400), nn.ReLU(),
            nn.Linear(400, latent_dim * 2)
        )
        self.decoder_fc = nn.Linear(latent_dim + K, 400)
        self.decoder = nn.Sequential(
            nn.ReLU(), nn.Linear(400, 784), nn.Sigmoid()
        )
    def encode(self, y):
        h = self.encoder(y)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z, x_prob):
        zx = torch.cat([z, x_prob], dim=1)
        h = self.decoder_fc(zx)
        return self.decoder(h).view(-1, 1, 28, 28)

# ==========================================
# 2. HMM
# ==========================================
class HMM_VAE(nn.Module):
    def __init__(self, K, latent_dim=16, device='cpu'):
        super().__init__()
        self.K = K
        self.device = device
        self.vae = ConditionalVAE(K, latent_dim)
        
        # # === 初始化：白纸 + 微弱噪声 ===（一定保留）
        # self.log_pi = nn.Parameter(torch.zeros(K))
        # init_log_A = torch.zeros(K, K) 
        # init_log_A += torch.randn(K, K) * 0.01 
        # init_log_A.fill_diagonal_(-1e9)
        # self.log_A = nn.Parameter(init_log_A)

        # 不要用 zeros 了！
        # self.log_pi = nn.Parameter(torch.zeros(K))
        # init_log_A = torch.zeros(K, K) 
        
        # === 10分类专用: 极端高方差初始化 ===
        # === 1. 初始化：改回弱噪声 ===
        self.log_pi = nn.Parameter(torch.zeros(K))
        # 用微弱的噪声，打破对称性即可，不要强加偏见
        init_log_A = torch.randn(K, K) * 1.0
        init_log_A.fill_diagonal_(-1e9)
        self.log_A = nn.Parameter(init_log_A)


    def get_hmm_params(self):
        pi = torch.softmax(self.log_pi, dim=0)
        A = torch.softmax(self.log_A, dim=1)
        return pi, A

    def compute_log_bk(self, y_batch):
        B, T, _, _, _ = y_batch.size()
        y_flat = y_batch.view(B * T, 1, 28, 28)
        mu, logvar = self.vae.encode(y_flat)
        z = self.vae.reparameterize(mu, logvar)
        
        log_bk = []
        for k in range(self.K):
            x_k = torch.zeros(B * T, self.K, device=self.device)
            x_k[:, k] = 1.0
            y_rec = self.vae.decode(z, x_k)
            recon = F.binary_cross_entropy(y_rec.view(B*T, -1), y_flat.view(B*T, -1), reduction='none').sum(1)
            
            # === 关键 Scaling ===
            recon = recon / 100.0  
            log_bk.append(-recon.view(B, T))
            
        return torch.stack(log_bk, dim=2), z, mu, logvar

    def forward_backward(self, log_bk):
        B, T, K = log_bk.size()
        pi, A = self.get_hmm_params()
        log_pi = torch.log(pi + 1e-9); log_A = torch.log(A + 1e-9)

        # Forward
        log_alpha_list = []
        log_alpha_0 = log_pi + log_bk[:, 0, :]
        log_alpha_list.append(log_alpha_0)
        for t in range(1, T):
            prev = log_alpha_list[-1]
            curr = torch.logsumexp(prev.unsqueeze(2) + log_A.unsqueeze(0), dim=1) + log_bk[:, t, :]
            log_alpha_list.append(curr)
        log_alpha = torch.stack(log_alpha_list, dim=1)

        # Backward
        log_beta_list = [torch.zeros(B, K, device=self.device)]
        for t in range(T-2, -1, -1):
            next_beta = log_beta_list[0]
            curr = torch.logsumexp(log_A.unsqueeze(0) + log_bk[:, t+1, :].unsqueeze(1) + next_beta.unsqueeze(1), dim=2)
            log_beta_list.insert(0, curr)
        log_beta = torch.stack(log_beta_list, dim=1)

        # Gamma & Xi
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=2, keepdim=True)
        
        xi_list = []
        for t in range(T-1):
            log_xi = log_alpha[:, t, :].unsqueeze(2) + log_A.unsqueeze(0) + log_bk[:, t+1, :].unsqueeze(1) + log_beta[:, t+1, :].unsqueeze(1)
            log_xi = log_xi - torch.logsumexp(log_xi.view(B, -1), dim=1).view(B, 1, 1)
            xi_list.append(torch.exp(log_xi))
            
        return torch.exp(log_gamma), torch.stack(xi_list, dim=1)

# ==========================================
# 3. 修正后的评估函数 (Brute Force)
# ==========================================
def evaluate_alignment_brute_force(est_A, true_A):
    K = true_A.shape[0]
    best_err = float('inf')
    best_perm = None
    
    # 尝试所有可能的排列 (K=3 时只有 6 种，K=10 时太慢，但这里 K=3)
    for perm in permutations(range(K)):
        perm = np.array(perm)
        # 重排矩阵: A_new = P * A * P.T
        # 对应 numpy 操作: est_A[perm][:, perm]
        # 这意味着我们将 est_state i 映射到 true_state perm[i]
        permuted_A = est_A[perm][:, perm]
        
        err = np.mean(np.abs(permuted_A - true_A))
        if err < best_err:
            best_err = err
            best_perm = perm
            
    return best_err, best_perm

from itertools import permutations

def get_best_alignment(est_A, true_A):
    """
    尝试所有可能的排列 (K=3 时只有 6 种情况)，找到误差最小的排列。
    返回：最佳误差，最佳排列索引
    """
    K = true_A.shape[0]
    best_err = float('inf')
    best_perm = None
    
    # 暴力尝试所有排列
    for perm in permutations(range(K)):
        perm = np.array(perm)
        # 重排矩阵: 行和列都要按 perm 重排
        # permuted_A[i, j] = est_A[perm[i], perm[j]]
        permuted_A = est_A[perm][:, perm]
        
        err = np.mean(np.abs(permuted_A - true_A))
        
        if err < best_err:
            best_err = err
            best_perm = perm
            
    return best_err, best_perm

from scipy.optimize import linear_sum_assignment

def evaluate_alignment_hungarian(est_A, true_A):
    """
    使用匈牙利算法进行 O(N^3) 的快速对齐，适用于 10 类及以上。
    注意：它可能无法完美处理“逆向循环”的误差计算，但能帮我们找到最佳可视化排列。
    """
    K = true_A.shape[0]
    # 构建 Cost Matrix
    cost_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            # 计算行相似度
            cost_matrix[i, j] = np.linalg.norm(est_A[i] - true_A[j])
    
    # 匈牙利算法求解最佳行列匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 根据结果重排矩阵
    # col_ind[i] 表示第 i 行应该去哪一列
    perm = col_ind 
    
    # 重排 est_A
    permuted_A = est_A[perm][:, perm]
    err = np.mean(np.abs(permuted_A - true_A))
    
    return err, perm

# ==========================================
# 4. 终极可视化 (矩阵数值 + 状态采样)
# ==========================================
def visualize_final_results(model, true_A, device):
    print("\n正在生成最终结果可视化...")
    
    # --- 1. 获取矩阵并对齐 ---
    est_A = model.get_hmm_params()[1].detach().cpu().numpy()
    # err, perm = evaluate_alignment_brute_force(est_A, true_A)
    err, perm = evaluate_alignment_hungarian(est_A, true_A)
    
    # 按最佳排列重组矩阵
    aligned_A = est_A[perm][:, perm]
    
    # --- 2. 绘制矩阵 (带概率数值) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # (左图) Ground Truth
    im0 = axes[0].imshow(true_A, vmin=0, vmax=1, cmap='Blues')
    axes[0].set_title("Ground Truth A", fontsize=14)
    # 循环标注概率值
    for (i, j), val in np.ndenumerate(true_A):
        color = 'white' if val > 0.5 else 'black'
        axes[0].text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=12)

    # (右图) Learned (Aligned)
    im1 = axes[1].imshow(aligned_A, vmin=0, vmax=1, cmap='Greens')
    axes[1].set_title(f"Learned A (Aligned)\nError: {err:.4f}", fontsize=14)
    # 循环标注概率值
    for (i, j), val in np.ndenumerate(aligned_A):
        color = 'white' if val > 0.5 else 'black'
        axes[1].text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("final_matrix_with_probs.png")
    print(f" [1/2] 矩阵图已保存: final_matrix_with_probs.png (Error={err:.4f})")
    
    # --- 3. 状态采样验证 (Sampling) ---
    model.eval()
    K = model.K
    # 反推 latent_dim (从 decoder 输入层维度减去 K)
    latent_dim = model.vae.decoder_fc.in_features - K
    samples_per_state = 10
    
    fig, axes = plt.subplots(K, samples_per_state, figsize=(samples_per_state, K*1.5))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    print(f" [2/2] 正在从状态采样生成图像 (Latent Dim={latent_dim})...")
    
    with torch.no_grad():
        for k in range(K): # 遍历每一个学习到的状态
            # 1. 构造 One-hot 条件向量
            x = torch.zeros(samples_per_state, K).to(device)
            x[:, k] = 1.0
            
            # 2. 从标准正态分布采样 z ~ N(0, 1)
            z = torch.randn(samples_per_state, latent_dim).to(device)
            
            # 3. 解码生成图像
            gen_imgs = model.vae.decode(z, x).cpu()
            
            # 4. 找到它对应的真实状态 (用于显示标签)
            # perm[k] 意思是: Learned State k 对应 True State perm[k]
            true_state_idx = perm[k]
            
            for i in range(samples_per_state):
                ax = axes[k, i]
                ax.imshow(gen_imgs[i].view(28, 28), cmap='gray')
                ax.axis('off')
                
                # 在每行的第一张图左边标注状态名
                if i == 0:
                    ax.text(-5, 14, f"Learned S{k}\n(True S{true_state_idx})", 
                            fontsize=12, va='center', ha='right', fontweight='bold')
                    
    plt.suptitle(f"Generated Samples from Learned States\n(Permutation: {perm})", fontsize=16)
    plt.savefig("final_state_samples_10_class.png")
    print(" [完成] 采样验证图已保存: final_state_samples_10_class.png")

def check_vae_reconstruction(model, loader, device):
    print("\n[Debug] 正在检查 VAE 的重构能力 (Reconstruction Check)...")
    model.eval()
    
    # 1. 拿一个 Batch 的真数据
    try:
        y_batch, = next(iter(loader))
    except ValueError:
        y_batch = next(iter(loader))[0]
        
    y_batch = y_batch.to(device)
    
    # 只取前 10 个序列
    n_samples = 10
    
    # === [关键修正] ===
    # y_batch 是 [B, T, 1, 28, 28]，包含了时间维度 T
    # 我们只取每个序列的"第0帧"来做重构测试
    real_imgs = y_batch[:n_samples, 0].view(n_samples, 1, 28, 28)
    
    # 2. 让 VAE 编码 -> 解码
    with torch.no_grad():
        # Encoder: y -> z
        # 注意：这里我们只把单帧图片送进去
        mu, logvar = model.vae.encode(real_imgs.view(n_samples, -1))
        z = model.vae.reparameterize(mu, logvar)
        
        # Decoder: z + (所有可能的 State) -> y_rec
        K = model.K
        
        fig, axes = plt.subplots(n_samples, K + 1, figsize=(K + 1, n_samples))
        
        for i in range(n_samples):
            # 第一列：画原图
            axes[i, 0].imshow(real_imgs[i].cpu().view(28, 28), cmap='gray')
            axes[i, 0].axis('off')
            if i == 0: axes[i, 0].set_title("Real", fontsize=10)
            
            # 后面的列：画各个 State 下的重构图
            for k in range(K):
                # 构造 One-hot State 向量
                x_k = torch.zeros(1, K).to(device)
                x_k[:, k] = 1.0
                
                # 解码
                # z[i:i+1] 是当前这张图的潜变量
                z_i = z[i:i+1] 
                rec_img = model.vae.decode(z_i, x_k)
                
                axes[i, k+1].imshow(rec_img.cpu().view(28, 28), cmap='gray')
                axes[i, k+1].axis('off')
                if i == 0: axes[i, k+1].set_title(f"S{k}", fontsize=10)
    
    plt.tight_layout()
    plt.savefig("debug_vae_reconstruction.png")
    print("[Debug] 重构对比图已保存: debug_vae_reconstruction.png")

def visualize_decoder_control(model, device, epoch):
    """
    画图验证：固定 z，改变 x，看图片是否变化。
    如果每一行的图片随着列变化而变化，说明 FiLM 生效了。
    """
    model.eval()
    K = model.K
    latent_dim = model.vae.decoder_fc.in_features - K
    n_samples = 5
    
    # 随机采样几个 z (Fixed style)
    z_fixed = torch.randn(n_samples, latent_dim).to(device)
    
    fig, axes = plt.subplots(n_samples, K, figsize=(K, n_samples))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    with torch.no_grad():
        for i in range(n_samples):
            for k in range(K):
                # 构造 One-hot x
                x_input = torch.zeros(1, K).to(device)
                x_input[0, k] = 1.0
                
                # Decode: Fixed z[i] + Condition x[k]
                img = model.vae.decode(z_fixed[i:i+1], x_input)
                
                ax = axes[i, k]
                ax.imshow(img.view(28, 28).cpu(), cmap='gray')
                ax.axis('off')
                if i == 0:
                    ax.set_title(f"S{k}", fontsize=8)
                if k == 0:
                    ax.text(-5, 14, f"z_{i}", va='center', fontsize=8)
    
    plt.suptitle(f"Decoder Control Check (Epoch {epoch})", fontsize=12)
    plt.savefig(f"control_check_ep{epoch:03d}.png")
    plt.close()
    print(f"Saved control_check_ep{epoch:03d}.png (Check this to see if x controls output!)")


def train_experiment():
    DATA_FILE = "hmm_mnist_data.pt"
    # DATA_FILE = "hmm_mnist_10_class.pt"
    if not os.path.exists(DATA_FILE): print("Data file not found."); return

    checkpoint = torch.load(DATA_FILE)
    sequences = checkpoint['sequences'].float()
    true_A = checkpoint['true_A']
    K = true_A.shape[0]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on", DEVICE)

    # === 配置 ===
    BATCH_SIZE = 128
    EPOCHS = 100
    LATENT_DIM = 3
    LR = 1e-3

    # 关键：预热策略
    WARMUP_EPOCHS = 0 # 前10轮只练 VAE
    TARGET_TRANS_WEIGHT = 2.0 

    dataset = TensorDataset(sequences)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = HMM_VAE(K, LATENT_DIM, device=DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        # === 动态权重 ===
        if epoch < WARMUP_EPOCHS:
            trans_weight = 0.0   # 预热期：HMM 不参与，VAE 专心看图
        else:
            trans_weight = TARGET_TRANS_WEIGHT # 启动 HMM
        total_loss_val = 0; emit_loss_val = 0; trans_loss_val = 0
        # === 新增: Warmup 结束时的中途检查 ===
        # 在 Warmup 的最后一轮 (例如第 9 轮) 结束时，画一张图看看
        if epoch == WARMUP_EPOCHS - 1:
            print(f"\n[Check] Warmup 结束 (Ep {epoch})! 正在检查 VAE 学习情况...")
            
            model.eval()
            K = model.K
            # 反推 latent_dim
            latent_dim = model.vae.decoder_fc.in_features - K
            
            # 画 10x10 的采样图
            fig, axes = plt.subplots(K, 10, figsize=(10, 10))
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            
            with torch.no_grad():
                for k in range(K):
                    # 构造 One-hot
                    x = torch.zeros(10, K).to(DEVICE)
                    x[:, k] = 1.0
                    # 采样 z
                    z = torch.randn(10, latent_dim).to(DEVICE)
                    # 解码
                    gen_imgs = model.vae.decode(z, x).cpu()
                    
                    for i in range(10):
                        ax = axes[k, i]
                        ax.imshow(gen_imgs[i].view(28, 28), cmap='gray')
                        ax.axis('off')
                        if i == 0:
                            ax.text(-5, 14, f"State {k}", fontsize=12, va='center', ha='right')
                            
            plt.suptitle(f"VAE Status after Warmup (Epoch {epoch})", fontsize=16)
            plt.savefig(f"check_warmup_ep{epoch}.png")
            print(f"[Check] 检查图已保存: check_warmup_ep{epoch}.png\n")

            check_vae_reconstruction(model, loader, DEVICE)
            
            model.train() # 切记改回训练模式

        model.train()
        for (y_batch,) in loader:
            y_batch = y_batch.to(DEVICE)
            B, T, _, _, _ = y_batch.size()
            optimizer.zero_grad()

            log_bk, z, mu, logvar = model.compute_log_bk(y_batch)
            with torch.no_grad():
                gamma, xi = model.forward_backward(log_bk)
            gamma = gamma.detach(); xi = xi.detach()

            weighted_recon = -torch.sum(gamma * log_bk) / (B*T)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (B*T)
            loss_emission = weighted_recon + kld

            pi, A = model.get_hmm_params()
            log_pi = torch.log(pi + 1e-9); log_A = torch.log(A + 1e-9)
            loss_init = -torch.sum(gamma[:, 0, :] * log_pi)
            loss_trans = -torch.sum(xi * log_A.unsqueeze(0).unsqueeze(0))
            loss_transition = (loss_init + loss_trans) / (B*T)

            loss = loss_emission + TARGET_TRANS_WEIGHT * loss_transition
            loss.backward(); optimizer.step()

            total_loss_val += loss.item(); emit_loss_val += loss_emission.item(); trans_loss_val += loss_transition.item()

        if epoch % 1 == 0:
            with torch.no_grad():
                est_A = model.get_hmm_params()[1].cpu().numpy()
                # 使用 Brute Force 评估
                # err, perm = evaluate_alignment_brute_force(est_A, true_A)
                err, perm = evaluate_alignment_hungarian(est_A, true_A)

           
            print(f"Ep {epoch:03d} | Emit {emit_loss_val/len(loader):.2f} | Trans {trans_loss_val/len(loader):.2f} | A_err {err:.4f}")
            
        # Visualization Check
        if epoch % 10 == 0:
            visualize_decoder_control(model, DEVICE, epoch)

    print("\nTraining Finished.")
    # 1. 获取学到的原始矩阵
    est_A = model.get_hmm_params()[1].detach().cpu().numpy()
    
    # 2. 使用【暴力对齐】找到最佳视角
    err, perm = get_best_alignment(est_A, true_A)
    
    # 3. 根据最佳排列重组矩阵
    # perm 是最佳的索引顺序，比如 [0, 2, 1]
    aligned_A = est_A[perm][:, perm]
    
    print(f"Best Alignment Error: {err:.4f}")
    print(f"Best Permutation: {perm}")
    
    # 4. 绘图对比
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # 左图：真值
    im0 = axes[0].imshow(true_A, vmin=0, vmax=1)
    axes[0].set_title("Ground Truth A")
    for (j,i),label in np.ndenumerate(true_A):
        axes[0].text(i,j,f"{label:.2f}",ha='center',va='center',color='red')
        
    # 右图：对齐后的学习矩阵
    im1 = axes[1].imshow(aligned_A, vmin=0, vmax=1)
    axes[1].set_title(f"Learned A (Aligned, Err={err:.3f})")
    for (j,i),label in np.ndenumerate(aligned_A):
        axes[1].text(i,j,f"{label:.2f}",ha='center',va='center',color='red')
    
    plt.tight_layout()
    plt.savefig("aligned_result_10_class.png")
    print("Saved aligned visualization to aligned_result_10_class.png")

    visualize_final_results(model, true_A, DEVICE)
    # 5. 打印状态映射关系
    # 如果 perm 是 [0, 2, 1]，意思是：
    # 学到的 State 0 对应 真实的 State 0
    # 学到的 State 1 对应 真实的 State 2
    # 学到的 State 2 对应 真实的 State 1
    print("\nState Mapping (Learned -> True):")
    for learned_idx, true_idx in enumerate(perm):
        print(f"Learned State {learned_idx}  ==>  True State {true_idx}")

if __name__ == "__main__":
    train_experiment()