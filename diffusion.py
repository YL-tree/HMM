"""
HMM-Diffusion for 10-class MNIST sequences

将 HMM-VAE 中的 VAE 替换为条件 Diffusion (DDPM)

显存优化设计:
  VAE: compute_log_bk 只需 K 次轻量 decoder, 可带梯度
  Diff: compute_log_bk 需要 K × n_mc 次 UNet forward, 必须 no_grad
  
  解决方案:
  1. compute_log_bk 全部 @torch.no_grad — 只用于状态分配
  2. denoiser 梯度只通过 compute_diffusion_loss 传 (单次 UNet forward)
  3. 预训练 balance loss: log_bk 无梯度, balance 仅监控
     MNIST 10数字本就均匀, argmax bootstrap 自然趋向均衡
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.optimize import linear_sum_assignment
from torchvision import datasets, transforms


# =============================================================
# Part 0: 数据生成
# =============================================================
def generate_hmm_mnist_10class(
    num_sequences=10000,
    seq_length=32,
    save_path="hmm_mnist_10_class.pt"
):
    K = 10
    noise = 0.01
    true_A = np.full((K, K), noise)
    for i in range(K):
        true_A[i, (i + 1) % K] = 1.0 - noise * (K - 1)
    true_A = true_A / true_A.sum(axis=1, keepdims=True)

    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    digit_images = {i: [] for i in range(K)}
    for img, label in mnist:
        if label < K:
            digit_images[label].append(img)

    pi = np.ones(K) / K
    sequences = []
    state_sequences = []  # 保存真实标签
    for _ in range(num_sequences):
        state = np.random.choice(K, p=pi)
        seq = []
        states = []
        for t in range(seq_length):
            digit = state
            states.append(state)
            idx = np.random.randint(len(digit_images[digit]))
            img = digit_images[digit][idx]
            seq.append(img)
            state = np.random.choice(K, p=true_A[state])
        sequences.append(torch.stack(seq))
        state_sequences.append(states)

    sequences = torch.stack(sequences)
    state_sequences = torch.tensor(state_sequences, dtype=torch.long)
    print(f"Generated {num_sequences} sequences, shape: {sequences.shape}")
    torch.save({'sequences': sequences, 'true_A': true_A, 'K': K,
                'state_sequences': state_sequences}, save_path)
    print(f"Saved to {save_path}")
    return true_A


# =============================================================
# Part 1: Diffusion 模型
# =============================================================
def get_beta_schedule(num_timesteps=100, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, num_timesteps)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ConditionalUNet(nn.Module):
    """条件 U-Net for 28x28, 3层, 通道 32/64/128
    每一层都注入条件（FiLM 调制: scale + shift）
    """
    def __init__(self, K, time_dim=64):
        super().__init__()
        self.K = K
        self.time_dim = time_dim
        cond_dim = time_dim * 2  # t_emb + x_emb 拼接

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(K, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder: 输入通道 = 1 (image) + K (condition broadcast)
        self.enc1 = self._conv_block(1 + K, 32)
        self.film_enc1 = nn.Linear(cond_dim, 32 * 2)   # scale + shift
        self.pool1 = nn.Conv2d(32, 32, 3, 2, 1)    # 28→14

        self.enc2 = self._conv_block(32, 64)
        self.film_enc2 = nn.Linear(cond_dim, 64 * 2)
        self.pool2 = nn.Conv2d(64, 64, 3, 2, 1)     # 14→7

        # Bottleneck
        self.bot = self._conv_block(64, 128)
        self.film_bot = nn.Linear(cond_dim, 128 * 2)

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)  # 7→14
        self.dec1 = self._conv_block(128, 64)
        self.film_dec1 = nn.Linear(cond_dim, 64 * 2)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)   # 14→28
        self.dec2 = self._conv_block(64, 32)
        self.film_dec2 = nn.Linear(cond_dim, 32 * 2)

        self.final = nn.Conv2d(32, 1, 1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def _apply_film(self, h, film_layer, cond):
        """FiLM: h = scale * h + shift"""
        params = film_layer(cond)           # [N, ch*2]
        scale, shift = params.chunk(2, dim=1)
        scale = scale[:, :, None, None]     # [N, ch, 1, 1]
        shift = shift[:, :, None, None]
        return h * (1 + scale) + shift

    def forward(self, y_noisy, t, x_state):
        t_emb = self.time_mlp(t)
        x_emb = self.state_mlp(x_state)
        cond = torch.cat([t_emb, x_emb], dim=1)  # [N, cond_dim]

        # Encoder (每层都注入条件)
        # 输入拼接: x_state [N,K] → [N,K,H,W] 和 y_noisy 拼接
        H, W = y_noisy.size(2), y_noisy.size(3)
        x_map = x_state[:, :, None, None].expand(-1, -1, H, W)
        inp = torch.cat([y_noisy, x_map], dim=1)  # [N, 1+K, H, W]

        e1 = self.enc1(inp)
        e1 = self._apply_film(e1, self.film_enc1, cond)

        e2 = self.enc2(self.pool1(e1))
        e2 = self._apply_film(e2, self.film_enc2, cond)

        # Bottleneck
        b = self.bot(self.pool2(e2))
        b = self._apply_film(b, self.film_bot, cond)

        # Decoder (每层都注入条件)
        d1 = self.up1(b)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        d1 = self._apply_film(d1, self.film_dec1, cond)

        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        d2 = self._apply_film(d2, self.film_dec2, cond)

        return self.final(d2)


class HMM_Diffusion(nn.Module):
    def __init__(self, K, num_timesteps=100, device='cpu'):
        super().__init__()
        self.K = K
        self.num_timesteps = num_timesteps
        self.device = device

        self.denoiser = ConditionalUNet(K)

        betas = get_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        self.log_pi = nn.Parameter(torch.zeros(K) * 0.1)
        init_log_A = torch.randn(K, K) * 0.1
        with torch.no_grad():
            init_log_A.fill_diagonal_(-5.0)
        self.log_A = nn.Parameter(init_log_A)
        self.log_scale = nn.Parameter(torch.tensor(0.0))

    def get_hmm_params(self):
        pi = torch.softmax(self.log_pi, dim=0)
        A = torch.softmax(self.log_A, dim=1)
        return pi, A

    def q_sample(self, y0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(y0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * y0 + sqrt_one_minus * noise, noise

    @torch.no_grad()
    def compute_log_bk(self, y_batch, scale_factor=5.0, n_mc=4):
        """
        log b_k(y) — 全部 no_grad
        对 n_mc 个随机时间步采样取平均, 近似条件对数似然
        
        诊断发现: 高 t 时刻 ratio(max-min_diff / avg_mse) 最大
        因为图像被噪声完全破坏后, denoiser 必须依赖条件 k
        t=[60,100): diff=0.01, ratio=34-45%, 信号最强
        """
        B, T_seq, C, H, W = y_batch.size()
        y_flat = y_batch.view(B * T_seq, C, H, W)
        N = B * T_seq
        scale = torch.exp(self.log_scale)
        
        # 高噪声段: denoiser 被迫依赖条件, 区分度最大
        t_lo = self.num_timesteps * 3 // 5   # 60
        t_hi = self.num_timesteps             # 100

        total_mse = torch.zeros(self.K, N, device=self.device)

        for _ in range(n_mc):
            t = torch.randint(t_lo, t_hi, (N,), device=self.device)
            noise = torch.randn_like(y_flat)
            y_noisy, _ = self.q_sample(y_flat, t, noise)

            for k in range(self.K):
                x_k = torch.zeros(N, self.K, device=self.device)
                x_k[:, k] = 1.0
                noise_pred = self.denoiser(y_noisy, t, x_k)
                mse = (noise_pred - noise).pow(2).view(N, -1).mean(1)
                total_mse[k] += mse

        avg_mse = total_mse / n_mc
        log_bk = (-avg_mse / scale_factor) * scale
        log_bk = log_bk.t().view(B, T_seq, self.K)
        return log_bk

    def compute_diffusion_loss(self, y_flat, x_state):
        """
        DDPM loss, 偏向高 t 采样以强化条件依赖
        
        标准 DDPM 均匀采样 t, denoiser 在低 t 不需要条件就能预测
        偏向高 t 后, denoiser 被迫更多依赖条件输入
        使用 beta 分布偏向高 t, 同时保留低 t 的覆盖
        """
        N = y_flat.size(0)
        # 用均匀分布的平方偏向高 t: u~U(0,1), t = u^0.5 * T
        # 这样 t 的分布密度在高 t 端更高
        u = torch.rand(N, device=self.device)
        t = (u.sqrt() * self.num_timesteps).long().clamp(0, self.num_timesteps - 1)
        noise = torch.randn_like(y_flat)
        y_noisy, _ = self.q_sample(y_flat, t, noise)
        noise_pred = self.denoiser(y_noisy, t, x_state)
        return F.mse_loss(noise_pred, noise)

    def compute_contrastive_loss(self, y_flat, x_state):
        """
        对比 loss: 鼓励正确条件的 MSE 低于随机错误条件的 MSE
        - 正确条件: noise_pred_pos = denoiser(y_noisy, t, x_correct)
        - 错误条件: noise_pred_neg = denoiser(y_noisy, t, x_random_wrong)
        - loss = max(0, MSE_pos - MSE_neg + margin)
        """
        N = y_flat.size(0)
        t = torch.randint(0, self.num_timesteps, (N,), device=self.device)
        noise = torch.randn_like(y_flat)
        y_noisy, _ = self.q_sample(y_flat, t, noise)

        # 正确条件
        noise_pred_pos = self.denoiser(y_noisy, t, x_state)
        mse_pos = (noise_pred_pos - noise).pow(2).view(N, -1).mean(1)  # [N]

        # 错误条件: 随机 shuffle 条件
        perm = torch.randperm(N, device=self.device)
        x_neg = x_state[perm]
        # 确保 x_neg != x_state (如果碰巧一样就再 shuffle)
        same = (x_neg.argmax(1) == x_state.argmax(1))
        while same.any():
            perm2 = torch.randperm(same.sum().item(), device=self.device)
            x_neg[same] = x_neg[same][perm2]
            same = (x_neg.argmax(1) == x_state.argmax(1))

        noise_pred_neg = self.denoiser(y_noisy, t, x_neg)
        mse_neg = (noise_pred_neg - noise).pow(2).view(N, -1).mean(1)  # [N]

        # Triplet margin loss: MSE_pos should be lower than MSE_neg by margin
        margin = 0.01
        contrastive = F.relu(mse_pos - mse_neg + margin).mean()

        return contrastive

    @torch.no_grad()
    def sample(self, x_state, n_samples=1):
        y = torch.randn(n_samples, 1, 28, 28, device=self.device)
        for t_idx in reversed(range(self.num_timesteps)):
            t = torch.full((n_samples,), t_idx, device=self.device, dtype=torch.long)
            noise_pred = self.denoiser(y, t, x_state)
            beta_t = self.betas[t_idx]
            alpha_t = self.alphas[t_idx]
            alpha_bar_t = self.alphas_cumprod[t_idx]
            y = (1.0 / alpha_t.sqrt()) * (y - beta_t / (1.0 - alpha_bar_t).sqrt() * noise_pred)
            if t_idx > 0:
                y = y + beta_t.sqrt() * torch.randn_like(y)
        return y.clamp(0, 1)

    # ======== HMM 部分 ========
    def forward_pass(self, log_bk):
        B, T, K = log_bk.size()
        pi, A = self.get_hmm_params()
        log_pi = torch.log(pi + 1e-9)
        log_A = torch.log(A + 1e-9)

        log_alpha_list = [log_pi + log_bk[:, 0, :]]
        for t in range(1, T):
            prev = log_alpha_list[-1]
            curr = torch.logsumexp(
                prev.unsqueeze(2) + log_A.unsqueeze(0), dim=1
            ) + log_bk[:, t, :]
            log_alpha_list.append(curr)

        return torch.stack(log_alpha_list, dim=1), log_A

    def backward_sample_gumbel(self, log_alpha, log_A, tau=1.0):
        B, T, K = log_alpha.size()
        x_samples = []
        logits_n = log_alpha[:, T-1, :]
        logits_n = logits_n - logits_n.max(dim=1, keepdim=True)[0]
        x_samples.append(F.gumbel_softmax(logits_n, tau=tau, hard=False))

        for t in range(T-2, -1, -1):
            x_next = x_samples[-1].detach()
            trans_score = torch.matmul(x_next, log_A.t())
            logits_t = log_alpha[:, t, :] + trans_score
            logits_t = logits_t - logits_t.max(dim=1, keepdim=True)[0]
            x_samples.append(F.gumbel_softmax(logits_t, tau=tau, hard=False))

        return torch.stack(x_samples[::-1], dim=1)

    @torch.no_grad()
    def viterbi_decode(self, log_bk):
        B, T, K = log_bk.size()
        pi, A = self.get_hmm_params()
        log_pi = torch.log(pi + 1e-9)
        log_A = torch.log(A + 1e-9)

        delta = log_pi + log_bk[:, 0, :]
        psi = []
        for t in range(1, T):
            scores = delta.unsqueeze(2) + log_A.unsqueeze(0)
            max_scores, max_idx = scores.max(dim=1)
            delta = max_scores + log_bk[:, t, :]
            psi.append(max_idx)

        states = torch.zeros(B, T, dtype=torch.long, device=log_bk.device)
        states[:, T-1] = delta.argmax(dim=1)
        for t in range(T-2, -1, -1):
            states[:, t] = psi[t].gather(1, states[:, t+1].unsqueeze(1)).squeeze(1)
        return states


# =============================================================
# Part 2: 评估与可视化
# =============================================================
def evaluate_alignment(est_A, true_A):
    K = true_A.shape[0]
    best_err = float('inf')
    best_perm = None

    def try_perm(perm):
        nonlocal best_err, best_perm
        permuted_A = est_A[perm][:, perm]
        err = np.mean(np.abs(permuted_A - true_A))
        if err < best_err:
            best_err = err
            best_perm = list(perm)

    for shift in range(K):
        try_perm([(i + shift) % K for i in range(K)])
    for shift in range(K):
        try_perm([(K - i + shift) % K for i in range(K)])

    cost_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost_matrix[i, j] = (np.linalg.norm(est_A[i] - true_A[j]) +
                                 np.linalg.norm(est_A[:, i] - true_A[:, j]))
    _, col_ind = linear_sum_assignment(cost_matrix)
    try_perm(col_ind)

    successors = np.argmax(est_A, axis=1)
    for start in range(K):
        cycle = [start]; visited = {start}; curr = start; valid = True
        for _ in range(K - 1):
            nxt = successors[curr]
            if nxt in visited: valid = False; break
            cycle.append(nxt); visited.add(nxt); curr = nxt
        if valid and len(cycle) == K:
            perm = [0]*K
            for i in range(K): perm[cycle[i]] = i
            try_perm(perm)
            perm_rev = [0]*K
            for i in range(K): perm_rev[cycle[i]] = (K-i) % K
            try_perm(perm_rev)

    return best_err, best_perm


def visualize_decoder_control(model, device, epoch):
    model.eval()
    K = model.K
    n_samples = 5
    fig, axes = plt.subplots(K, n_samples, figsize=(n_samples*2, K*2))
    with torch.no_grad():
        for k in range(K):
            x_k = torch.zeros(n_samples, K, device=device)
            x_k[:, k] = 1.0
            samples = model.sample(x_k, n_samples=n_samples)
            for j in range(n_samples):
                axes[k, j].imshow(samples[j, 0].cpu().numpy(), cmap='gray')
                axes[k, j].axis('off')
                if j == 0:
                    axes[k, j].set_ylabel(f'S{k}', fontsize=10)
    plt.suptitle(f'Decoder Control (Epoch {epoch})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"decoder_control_diffusion_ep{epoch:03d}.png", dpi=120, bbox_inches='tight')
    plt.close()
    model.train()


def visualize_final_results(model, true_A, device):
    model.eval()
    K = model.K
    n_samples = 8
    est_A = model.get_hmm_params()[1].cpu().detach().numpy()
    _, perm = evaluate_alignment(est_A, true_A)

    fig, axes = plt.subplots(K, n_samples, figsize=(n_samples*1.5, K*1.5))
    with torch.no_grad():
        for row_idx in range(K):
            k = perm.index(row_idx) if row_idx in perm else row_idx
            x_k = torch.zeros(n_samples, K, device=device)
            x_k[:, k] = 1.0
            samples = model.sample(x_k, n_samples=n_samples)
            for j in range(n_samples):
                axes[row_idx, j].imshow(samples[j, 0].cpu().numpy(), cmap='gray')
                axes[row_idx, j].axis('off')
            axes[row_idx, 0].set_ylabel(f'S{k}\n(T{row_idx})', fontsize=8)
    plt.tight_layout()
    plt.savefig("final_samples_diffusion.png", dpi=150, bbox_inches='tight')
    plt.close()

    aligned_A = est_A[perm][:, perm]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(true_A, cmap='Blues', vmin=0, vmax=1)
    ax1.set_title("Ground Truth A")
    for i in range(K):
        for j in range(K):
            if true_A[i, j] > 0.1:
                ax1.text(j, i, f'{true_A[i,j]:.2f}', ha='center', va='center', fontsize=8)
    err_val = np.mean(np.abs(aligned_A - true_A))
    ax2.imshow(aligned_A, cmap='Greens', vmin=0, vmax=1)
    ax2.set_title(f"Learned A (Aligned)\nError: {err_val:.4f}")
    for i in range(K):
        for j in range(K):
            if aligned_A[i, j] > 0.1:
                ax2.text(j, i, f'{aligned_A[i,j]:.2f}', ha='center', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig("A_matrix_comparison_diffusion.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved visualizations.")


def check_data_stickiness(data_file):
    checkpoint = torch.load(data_file, weights_only=False)
    seq0 = checkpoint['sequences'][0]
    n_show = min(seq0.size(0), 16)
    plt.figure(figsize=(n_show*1.2, 1.5))
    for t in range(n_show):
        plt.subplot(1, n_show, t+1)
        plt.imshow(seq0[t].view(28, 28), cmap='gray')
        plt.title(f"t={t}", fontsize=6); plt.axis('off')
    plt.savefig("real_data_check_10class.png", dpi=120, bbox_inches='tight')
    plt.close()


# =============================================================
# Part 3: 训练
# =============================================================
def train_experiment():
    DATA_FILE = "hmm_mnist_10_class.pt"
    BATCH_SIZE = 64

    if not os.path.exists(DATA_FILE):
        print("=" * 50)
        print("Generating 10-class HMM-MNIST data...")
        print("=" * 50)
        generate_hmm_mnist_10class(num_sequences=10000, seq_length=32, save_path=DATA_FILE)

    checkpoint = torch.load(DATA_FILE, weights_only=False)
    sequences = checkpoint['sequences'].float()
    true_A = checkpoint['true_A']
    K = true_A.shape[0]
    true_states = checkpoint.get('state_sequences', None)  # 真实标签 (如果有)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {DEVICE}, K={K}")
    print(f"Data shape: {sequences.shape}")
    if true_states is not None:
        print(f"True states available: {true_states.shape}")
    num_seq = sequences.size(0)
    seq_len = sequences.size(1)

    dataset = TensorDataset(sequences)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 带 index 的 loader (用于缓存分配)
    indices_tensor = torch.arange(num_seq)
    indexed_dataset = TensorDataset(sequences, indices_tensor)
    indexed_loader = DataLoader(indexed_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ==========================================
    # 超参数
    # ==========================================
    NUM_TIMESTEPS = 100
    TAU_START = 2.0
    TAU_MIN = 0.1

    N_MC_ESTEP = 16             # E-step MC 采样数 (no_grad, 可以多一些)
    N_MC_LOG_BK = 8             # HMM 阶段每 batch MC 采样数
    REASSIGN_BATCH = 128

    # ==========================================
    # 严格 EM 预训练: 交替 E-step / M-step
    #   E-step: 全局 compute_log_bk → argmax 分配
    #   M-step: 固定分配, 跑若干 epoch 的条件 DDPM
    #   第一轮多训练 (denoiser 从零开始, 需要充分学习 K-means 分配)
    #   后续轮少训练, 频繁校正, 避免过拟合到错误分配
    # ==========================================
    N_EM_ROUNDS = 8
    M_EPOCHS_FIRST = 15         # 第一轮: 从零开始, 需要充分训练
    M_EPOCHS_REST = 5           # 后续轮: 频繁校正
    # 总预训练 = 15 + 7 * 5 = 50

    PRETRAIN_TOTAL = M_EPOCHS_FIRST + (N_EM_ROUNDS - 1) * M_EPOCHS_REST  # 50
    HMM_WARMUP_END = PRETRAIN_TOTAL + 50     # 100
    TOTAL_EPOCHS = PRETRAIN_TOTAL + 100      # 150

    LOAD_PRETRAINED = False
    PRETRAIN_PATH = "checkpoint_pretrain_diffusion.pt"

    model = HMM_Diffusion(K, num_timesteps=NUM_TIMESTEPS, device=DEVICE).to(DEVICE)

    denoiser_params = [p for n, p in model.named_parameters()
                       if 'log_A' not in n and 'log_pi' not in n]
    hmm_params = [model.log_A, model.log_pi]
    optimizer = optim.Adam([
        {'params': denoiser_params, 'lr': 2e-4},
        {'params': hmm_params, 'lr': 1e-2}
    ])

    start_epoch = 0
    if LOAD_PRETRAINED and os.path.exists(PRETRAIN_PATH):
        print(f"Loading pretrained model from {PRETRAIN_PATH}...")
        try:
            ckpt = torch.load(PRETRAIN_PATH, weights_only=False)
            model.load_state_dict(ckpt)
            print(">> Model weights loaded.")
            with torch.no_grad():
                model.log_pi.data.copy_(torch.randn(K).to(DEVICE) * 0.1)
                init_log_A = torch.randn(K, K).to(DEVICE) * 0.1
                init_log_A.fill_diagonal_(-5.0)
                model.log_A.data.copy_(init_log_A)
                model.log_scale.data.fill_(0.0)
            start_epoch = PRETRAIN_TOTAL
            print(f">> Starting from Epoch {start_epoch}.")
        except Exception as e:
            print(f"!! Failed to load: {e}. Starting from scratch.")
            start_epoch = 0
    else:
        print("Starting training from scratch...")

    # ==========================================
    # 全局状态分配缓存 — K-means 初始化打破对称性
    # ==========================================
    print("Initializing assignments with K-means...")
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    # 把所有图像 flatten 成向量
    all_images = sequences.view(num_seq * seq_len, -1).numpy()  # [N, 784]

    # PCA 降到 50 维再聚类 (快且稳定)
    pca = PCA(n_components=50, random_state=42)
    features = pca.fit_transform(all_images)
    print(f"  PCA: {all_images.shape} -> {features.shape}, "
          f"explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(features)  # [N]

    # 统计 k-means 分配质量
    cached_assignments = torch.tensor(labels, dtype=torch.long).view(num_seq, seq_len)
    km_freq = torch.bincount(cached_assignments.view(-1), minlength=K).float()
    km_freq = km_freq / km_freq.sum()
    print(f"  K-means done: freq=[{km_freq.min():.3f}-{km_freq.max():.3f}]")
    print(f"  K-means cluster sizes: {torch.bincount(cached_assignments.view(-1), minlength=K).tolist()}")

    # 验证: k-means 给每个数字的分配是否有倾向性
    # (如果数据有真实标签可以对比, 这里只看分布)
    for k_idx in range(K):
        mask = (cached_assignments.view(-1) == k_idx)
        imgs_k = all_images[mask.numpy()]
        mean_intensity = imgs_k.mean()
        print(f"  Cluster {k_idx}: n={mask.sum().item()}, mean_pixel={mean_intensity:.4f}")

    print("Start Training...")
    best_err = float('inf')

    # ==========================================
    # 辅助函数: E-step (全局重算分配)
    # ==========================================
    def do_estep(scale_factor):
        """E-step: 全局重算分配, 返回 (assignments, gap, n_used, freq, all_log_bk)"""
        model.eval()
        reassign_loader = DataLoader(
            TensorDataset(sequences, indices_tensor),
            batch_size=REASSIGN_BATCH, shuffle=False)
        all_log_bk = []
        with torch.no_grad():
            for y_b, idx_b in reassign_loader:
                y_b = y_b.to(DEVICE)
                lbk = model.compute_log_bk(y_b, scale_factor=scale_factor,
                                           n_mc=N_MC_ESTEP)
                all_log_bk.append(lbk.cpu())
        all_log_bk = torch.cat(all_log_bk, dim=0)  # [num_seq, seq_len, K]

        assignments = all_log_bk.argmax(dim=2)
        vals, _ = torch.topk(all_log_bk, 2, dim=2)
        gap = (vals[:, :, 0] - vals[:, :, 1]).mean().item()
        n_used = len(assignments.unique())
        freq = torch.bincount(assignments.view(-1), minlength=K).float()
        freq = freq / freq.sum()
        model.train()
        return assignments, gap, n_used, freq.numpy(), all_log_bk

    # ==========================================
    # Phase 1: 严格 EM 预训练
    # ==========================================
    global_epoch = 0
    for em_round in range(N_EM_ROUNDS):
        sf = 0.01
        cur_m_epochs = M_EPOCHS_FIRST if em_round == 0 else M_EPOCHS_REST

        # --- E-step ---
        print(f"\n{'='*60}")
        print(f"EM Round {em_round+1}/{N_EM_ROUNDS} | E-step (scale_factor={sf})")
        print(f"{'='*60}")

        if em_round == 0:
            # Round 1: 跳过 E-step, 直接用 K-means 初始分配
            gap = 0.0
            n_used = len(cached_assignments.unique())
            freq_t = torch.bincount(cached_assignments.view(-1), minlength=K).float()
            freq_np = (freq_t / freq_t.sum()).numpy()
            print(f"  [Skip E-step, using K-means init]")
            print(f"  #st={n_used} | freq=[{freq_np.min():.3f}-{freq_np.max():.3f}]")
        else:
            # Round 2+: 保守更新 — 只更新高置信度的样本, 保护已有分配
            new_assignments, gap, n_used, freq_np, all_log_bk_tmp = do_estep(sf)

            # 每个样本的置信度 = top1 - top2 gap
            vals_tmp, _ = torch.topk(all_log_bk_tmp, 2, dim=2)
            per_sample_gap = vals_tmp[:, :, 0] - vals_tmp[:, :, 1]  # [num_seq, seq_len]

            # 只更新 gap > median 的样本
            median_gap = per_sample_gap.median().item()
            confident_mask = (per_sample_gap > median_gap)
            update_ratio = confident_mask.float().mean().item()
            cached_assignments[confident_mask] = new_assignments[confident_mask]

            # 重算更新后的统计
            freq_t = torch.bincount(cached_assignments.view(-1), minlength=K).float()
            freq_np = (freq_t / freq_t.sum()).numpy()
            n_used = len(cached_assignments.unique())

            print(f"  Gap={gap:.4f} | #st={n_used} | "
                  f"freq=[{freq_np.min():.3f}-{freq_np.max():.3f}] | "
                  f"updated={update_ratio:.1%} (threshold={median_gap:.4f})")

        # --- M-step: 跑 cur_m_epochs 个 epoch ---
        print(f"  M-step: training {cur_m_epochs} epochs with fixed assignments...")
        for m_ep in range(cur_m_epochs):
            if global_epoch < start_epoch:
                global_epoch += 1
                continue

            optimizer.param_groups[0]['lr'] = 2e-4
            optimizer.param_groups[1]['lr'] = 0.0
            model.log_scale.requires_grad = False

            total_loss = 0
            model.train()
            for y_batch, idx_batch in indexed_loader:
                y_batch = y_batch.to(DEVICE)
                B, T, C, H, W = y_batch.size()
                y_flat = y_batch.view(B * T, C, H, W)
                N = B * T
                optimizer.zero_grad()

                assign = cached_assignments[idx_batch].long()
                x_state = F.one_hot(assign, num_classes=K).float().view(N, K).to(DEVICE)
                diff_loss = model.compute_diffusion_loss(y_flat, x_state)
                diff_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += diff_loss.item()

            avg_loss = total_loss / len(indexed_loader)

            with torch.no_grad():
                est_A = model.get_hmm_params()[1].cpu().numpy()
                err, perm = evaluate_alignment(est_A, true_A)
                if err < best_err:
                    best_err = err

            if m_ep == 0 or m_ep == cur_m_epochs - 1:
                print(f"  Ep {global_epoch:03d} [EM-R{em_round+1} M{m_ep+1:02d}] | "
                      f"Loss {avg_loss:.4f} | A_err {err:.4f} (best {best_err:.4f}) | sf={sf}")
            global_epoch += 1

        # --- 每个 Round 结束后: 画图 + 评估 ---
        visualize_decoder_control(model, DEVICE, global_epoch - 1)
        eval_assign, gap_after, n_used_after, freq_after, _ = do_estep(sf)
        diag_str = (f"  >> After R{em_round+1}: Gap={gap_after:.4f} | #st={n_used_after} | "
                    f"freq=[{freq_after.min():.3f}-{freq_after.max():.3f}]")
        if true_states is not None:
            cost = np.zeros((K, K))
            ea = eval_assign.view(-1).numpy()
            ts = true_states.view(-1).numpy()
            for ki in range(K):
                for kj in range(K):
                    cost[ki, kj] = -np.sum((ea == ki) & (ts == kj))
            row_ind, col_ind = linear_sum_assignment(cost)
            aligned = np.zeros_like(ea)
            for ki, kj in zip(row_ind, col_ind):
                aligned[ea == ki] = kj
            acc = np.mean(aligned == ts)
            diag_str += f" | Acc={acc:.3f}"
        print(diag_str)

        # --- R1 结束后: 诊断不同 t 范围的 MSE 差异 ---
        if em_round == 0:
            print("\n  [Diagnostic] MSE gap across timestep ranges:")
            model.eval()
            diag_y = sequences[:100].to(DEVICE)
            diag_flat = diag_y.view(-1, 1, 28, 28)
            N_diag = diag_flat.size(0)
            with torch.no_grad():
                for t_lo_d, t_hi_d in [(0,20), (20,40), (40,60), (60,80), (80,100)]:
                    mse_per_k = torch.zeros(K, N_diag, device=DEVICE)
                    for _ in range(8):
                        t_d = torch.randint(t_lo_d, t_hi_d, (N_diag,), device=DEVICE)
                        noise_d = torch.randn_like(diag_flat)
                        y_noisy_d, _ = model.q_sample(diag_flat, t_d, noise_d)
                        for k in range(K):
                            x_k_d = torch.zeros(N_diag, K, device=DEVICE)
                            x_k_d[:, k] = 1.0
                            pred_d = model.denoiser(y_noisy_d, t_d, x_k_d)
                            mse_d = (pred_d - noise_d).pow(2).view(N_diag, -1).mean(1)
                            mse_per_k[k] += mse_d
                    mse_per_k /= 8
                    # 每个样本: best_k 和 worst_k 的 MSE 差
                    mse_min = mse_per_k.min(dim=0)[0]
                    mse_max = mse_per_k.max(dim=0)[0]
                    diff = (mse_max - mse_min).mean().item()
                    avg_mse_val = mse_per_k.mean().item()
                    print(f"    t=[{t_lo_d:3d},{t_hi_d:3d}): avg_mse={avg_mse_val:.4f} | "
                          f"max-min_diff={diff:.5f} | ratio={diff/avg_mse_val:.5f}")
            model.train()

    # 保存预训练 checkpoint
    print(f"\n>> Pretrain done. Saving to {PRETRAIN_PATH}...")
    torch.save(model.state_dict(), PRETRAIN_PATH)

    # 最终 E-step 评估
    final_assign, final_gap, final_n_used, final_freq, _ = do_estep(0.01)
    print(f">> Final pretrain: Gap={final_gap:.4f} | #st={final_n_used}")
    if final_gap < 1.0:
        print(f">> WARNING: Gap={final_gap:.4f} < 1.0, 预训练可能不充分!")
    else:
        print(f">> Gap={final_gap:.4f} >= 1.0, 预训练成功!")
    print()

    # ==========================================
    # Phase 2/3: HMM 阶段 (也用 EM 缓存加速)
    # 每 HMM_REASSIGN_EVERY 个 epoch 做一次全局 E-step:
    #   log_bk → forward-backward → Gumbel 采样 → 缓存 soft 分配
    # 中间用缓存分配快速训练 denoiser + HMM 参数
    # ==========================================
    HMM_REASSIGN_EVERY = 5

    # 初始化 HMM soft 分配缓存 [num_seq, seq_len, K]
    cached_X_d = F.one_hot(cached_assignments, num_classes=K).float()

    for epoch in range(PRETRAIN_TOTAL, TOTAL_EPOCHS):
        hmm_epoch = epoch - PRETRAIN_TOTAL
        tau = max(TAU_MIN, TAU_START * (0.98 ** hmm_epoch))

        if epoch < HMM_WARMUP_END:
            phase_name = "HMM-Warmup"
            optimizer.param_groups[0]['lr'] = 0.0
            optimizer.param_groups[1]['lr'] = 1e-2
            current_scale_factor = 0.05
            model.log_scale.requires_grad = False
        else:
            phase_name = "Fine-tune"
            optimizer.param_groups[0]['lr'] = 1e-5
            optimizer.param_groups[1]['lr'] = 5e-2
            current_scale_factor = 0.05
            model.log_scale.requires_grad = False

        # --- HMM E-step: 每 R 个 epoch 全局重算 ---
        if hmm_epoch % HMM_REASSIGN_EVERY == 0:
            print(f"  [HMM E-step] epoch {epoch}, tau={tau:.3f}")
            model.eval()
            reassign_loader = DataLoader(
                TensorDataset(sequences, indices_tensor),
                batch_size=REASSIGN_BATCH, shuffle=False)

            all_X_d = []
            with torch.no_grad():
                for y_b, idx_b in reassign_loader:
                    y_b = y_b.to(DEVICE)
                    lbk = model.compute_log_bk(y_b, scale_factor=current_scale_factor,
                                               n_mc=N_MC_ESTEP)
                    log_alpha, log_A_val = model.forward_pass(lbk)
                    X = model.backward_sample_gumbel(log_alpha, log_A_val, tau=tau)
                    all_X_d.append(X.cpu())

            cached_X_d = torch.cat(all_X_d, dim=0)  # [num_seq, seq_len, K]

            # 统计
            hard = cached_X_d.argmax(dim=2)
            vals, _ = torch.topk(
                model.compute_log_bk(next(iter(loader))[0][:8].to(DEVICE),
                                     scale_factor=current_scale_factor, n_mc=N_MC_LOG_BK).cpu(),
                2, dim=2)
            gap_sample = (vals[:, :, 0] - vals[:, :, 1]).mean().item()
            hmm_n_used = len(hard.unique())
            print(f"  [HMM E-step] Gap≈{gap_sample:.2f} | #st={hmm_n_used} | tau={tau:.3f}")
            model.train()

        # --- HMM M-step: 用缓存分配训练 ---
        total_loss = 0; total_emit = 0; total_trans = 0
        model.train()

        for y_batch, idx_batch in indexed_loader:
            y_batch = y_batch.to(DEVICE)
            B, T, C, H, W = y_batch.size()
            y_flat = y_batch.view(B * T, C, H, W)
            N = B * T
            optimizer.zero_grad()

            # 从缓存取 soft 分配
            X_d = cached_X_d[idx_batch].to(DEVICE).detach()  # [B, T, K]
            x_state = X_d.view(N, K)

            diff_loss = model.compute_diffusion_loss(y_flat, x_state)

            pi, A = model.get_hmm_params()
            log_pi_safe = torch.log(pi + 1e-9)
            log_A_safe = torch.log(A + 1e-9)
            loss_init = -torch.sum(X_d[:, 0, :] * log_pi_safe) / B

            x_prev = X_d[:, :-1, :]
            x_curr = X_d[:, 1:, :]
            trans_score = torch.einsum('btj,btk,jk->', x_prev, x_curr, log_A_safe)
            loss_trans = -trans_score / (B * (T - 1))
            loss_transition = loss_init + loss_trans

            loss = diff_loss + loss_transition
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_emit += diff_loss.item()
            total_trans += loss_transition.item()

        avg_loss = total_loss / len(indexed_loader)
        avg_emit = total_emit / len(indexed_loader)
        avg_trans = total_trans / len(indexed_loader)
        current_scale = np.exp(model.log_scale.item())

        with torch.no_grad():
            est_A = model.get_hmm_params()[1].cpu().numpy()
            err, perm = evaluate_alignment(est_A, true_A)
            if err < best_err:
                best_err = err
                torch.save(model.state_dict(), "best_model_diffusion.pt")

        print(f"Ep {epoch:03d} [{phase_name:10s}] | Loss {avg_loss:.4f} | "
              f"Emit {avg_emit:.4f} | Trans {avg_trans:.4f} | "
              f"A_err {err:.4f} (best {best_err:.4f}) | "
              f"Scale {current_scale:.3f} | tau {tau:.3f}")

        if epoch % 20 == 0:
            visualize_decoder_control(model, DEVICE, epoch)
            model.eval()
            with torch.no_grad():
                sample_y = next(iter(loader))[0][:4].to(DEVICE)
                sample_log_bk = model.compute_log_bk(
                    sample_y, scale_factor=current_scale_factor, n_mc=N_MC_LOG_BK)
                viterbi_states = model.viterbi_decode(sample_log_bk)
                emission_states = sample_log_bk.argmax(dim=2)

                print(f"  Viterbi  (seq 0): {viterbi_states[0].tolist()}")
                print(f"  Emission (seq 0): {emission_states[0].tolist()}")

                v = viterbi_states[0].cpu().numpy()
                transitions = [(v[t], v[t+1]) for t in range(len(v)-1)]
                unique_trans = set(transitions)
                print(f"  Unique transitions ({len(unique_trans)}): "
                      f"{sorted(unique_trans)[:15]}{'...' if len(unique_trans)>15 else ''}")

            if perm is not None:
                aligned_A = est_A[perm][:, perm]
                diag_vals = [aligned_A[i, (i+1) % K] for i in range(K)]
                off_max = [np.max(np.delete(aligned_A[i], (i+1) % K)) for i in range(K)]
                print(f"  A[i, i+1] (should be ~0.9): "
                      f"mean={np.mean(diag_vals):.3f} min={np.min(diag_vals):.3f}")
                print(f"  max off-target per row:     "
                      f"mean={np.mean(off_max):.3f} max={np.max(off_max):.3f}")
            model.train()

    model.load_state_dict(torch.load("best_model_diffusion.pt", weights_only=False))
    visualize_final_results(model, true_A, DEVICE)
    print(f"\nTraining complete. Best A_err: {best_err:.4f}")


if __name__ == "__main__":
    DATA_FILE = "hmm_mnist_10_class.pt"
    if os.path.exists(DATA_FILE):
        check_data_stickiness(DATA_FILE)
    train_experiment()
