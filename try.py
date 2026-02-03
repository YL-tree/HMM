import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------
# Conditional ConvVAE
# 核心改动：使用 concatenation 而非 FiLM，更直接地注入 x
# ---------------------------------------------------------
class ConditionalConvVAE(nn.Module):
    def __init__(self, K, latent_dim=32):
        super().__init__()
        self.K = K
        self.latent_dim = latent_dim

        # Encoder: 不依赖 x，只从 y 提取 z
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_encode = nn.Linear(3136, latent_dim * 2)

        # Decoder: z 和 x 拼接后一起输入
        # 这是标准 cVAE 做法，比 FiLM 更直接
        self.fc_decode = nn.Linear(latent_dim + K, 512)
        self.fc_decode2 = nn.Linear(512, 3136)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, y):
        h = self.encoder_conv(y)
        h = self.fc_encode(h)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_prob):
        # 拼接 z 和 x
        zx = torch.cat([z, x_prob], dim=1)
        h = F.relu(self.fc_decode(zx))
        h = F.relu(self.fc_decode2(h))
        h = h.view(-1, 64, 7, 7)
        return self.decoder_conv(h)

# ---------------------------------------------------------
# HMM-VAE (Manuscript Section 2.3.1)
# ---------------------------------------------------------
class HMM_VAE(nn.Module):
    def __init__(self, K, latent_dim=32, device='cpu'):
        super().__init__()
        self.K = K
        self.device = device
        self.vae = ConditionalConvVAE(K, latent_dim)

        # HMM parameters
        self.log_pi = nn.Parameter(torch.zeros(K))
        self.log_A = nn.Parameter(torch.randn(K, K) * 0.1)

    def get_hmm_params(self):
        pi = torch.softmax(self.log_pi, dim=0)
        A = torch.softmax(self.log_A, dim=1)
        return pi, A

    def compute_log_bk(self, y_batch, scale_factor=50.0):
        """
        Compute log b_k(y, z) = log p(y | z, x=k, theta)

        scale_factor: 控制 log_bk 的数值范围
        - 太大 (如 200): log_bk 差异太小，状态难以区分
        - 太小 (如 10): log_bk 差异太大，Gumbel softmax 饱和
        - 推荐值: 30-100，使 gap 在 0.5-2.0 之间
        """
        B, T, _, _, _ = y_batch.size()
        y_flat = y_batch.view(B * T, 1, 28, 28)

        mu, logvar = self.vae.encode(y_flat)
        z = self.vae.reparameterize(mu, logvar)

        log_bk = []
        for k in range(self.K):
            x_k = torch.zeros(B * T, self.K, device=self.device)
            x_k[:, k] = 1.0
            y_rec = self.vae.decode(z, x_k)

            bce = F.binary_cross_entropy(
                y_rec.view(B * T, -1),
                y_flat.view(B * T, -1),
                reduction='none'
            ).sum(1)

            log_bk.append((-bce / scale_factor).view(B, T))

        log_bk = torch.stack(log_bk, dim=2)
        return log_bk, z, mu, logvar

    def forward_pass(self, log_bk):
        """Forward algorithm (Manuscript page 11)"""
        B, T, K = log_bk.size()
        pi, A = self.get_hmm_params()
        log_pi = torch.log(pi + 1e-9)
        log_A = torch.log(A + 1e-9)

        log_alpha_list = []
        log_alpha_0 = log_pi + log_bk[:, 0, :]
        log_alpha_list.append(log_alpha_0)

        for t in range(1, T):
            prev = log_alpha_list[-1]
            curr = torch.logsumexp(
                prev.unsqueeze(2) + log_A.unsqueeze(0), dim=1
            ) + log_bk[:, t, :]
            log_alpha_list.append(curr)

        log_alpha = torch.stack(log_alpha_list, dim=1)
        return log_alpha, log_A

    def backward_sample_gumbel(self, log_alpha, log_A, tau=1.0):
        """Backward sampling with Gumbel softmax (Manuscript page 12)"""
        B, T, K = log_alpha.size()
        x_samples = []

        # Sample x_n
        logits_n = log_alpha[:, T - 1, :]
        logits_n = logits_n - logits_n.max(dim=1, keepdim=True)[0]
        x_n = F.gumbel_softmax(logits_n, tau=tau, hard=False)
        x_samples.append(x_n)

        # Backward
        for t in range(T - 2, -1, -1):
            x_next = x_samples[-1]
            trans_score = torch.matmul(x_next, log_A.t())
            logits_t = log_alpha[:, t, :] + trans_score
            logits_t = logits_t - logits_t.max(dim=1, keepdim=True)[0]
            x_t = F.gumbel_softmax(logits_t, tau=tau, hard=False)
            x_samples.append(x_t)

        x_samples = x_samples[::-1]
        X = torch.stack(x_samples, dim=1)
        return X

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def evaluate_alignment_hungarian(est_A, true_A):
    K = true_A.shape[0]
    cost_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost_matrix[i, j] = np.linalg.norm(est_A[i] - true_A[j])
    _, col_ind = linear_sum_assignment(cost_matrix)
    perm = col_ind
    permuted_A = est_A[perm][:, perm]
    err = np.mean(np.abs(permuted_A - true_A))
    return err, perm

def visualize_decoder_control(model, device, epoch):
    """验证 x 是否控制输出：固定 z，改变 x"""
    model.eval()
    K = model.K
    latent_dim = model.vae.latent_dim
    n_samples = 5

    z_fixed = torch.randn(n_samples, latent_dim).to(device)

    fig, axes = plt.subplots(n_samples, K, figsize=(K * 1.5, n_samples * 1.5))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    with torch.no_grad():
        for i in range(n_samples):
            for k in range(K):
                x_input = torch.zeros(1, K).to(device)
                x_input[0, k] = 1.0
                img = model.vae.decode(z_fixed[i:i+1], x_input)

                if n_samples == 1:
                    ax = axes[k]
                else:
                    ax = axes[i, k]
                ax.imshow(img.view(28, 28).cpu(), cmap='gray')
                ax.axis('off')
                if i == 0:
                    ax.set_title(f"S{k}", fontsize=10)

    plt.suptitle(f"Decoder Control (Epoch {epoch})", fontsize=12)
    plt.savefig(f"control_check_ep{epoch:03d}.png", dpi=100, bbox_inches='tight')
    plt.close()

def visualize_final_results(model, true_A, device):
    est_A = model.get_hmm_params()[1].detach().cpu().numpy()
    err, perm = evaluate_alignment_hungarian(est_A, true_A)
    aligned_A = est_A[perm][:, perm]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(true_A, vmin=0, vmax=1, cmap='Blues')
    axes[0].set_title("Ground Truth A", fontsize=14)
    for (i, j), val in np.ndenumerate(true_A):
        color = 'white' if val > 0.5 else 'black'
        axes[0].text(j, i, f"{val:.2f}", ha='center', va='center', color=color)

    axes[1].imshow(aligned_A, vmin=0, vmax=1, cmap='Greens')
    axes[1].set_title(f"Learned A (Aligned)\nError: {err:.4f}", fontsize=14)
    for (i, j), val in np.ndenumerate(aligned_A):
        color = 'white' if val > 0.5 else 'black'
        axes[1].text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontweight='bold')
    plt.tight_layout()
    plt.savefig("conv_final_matrix.png")
    plt.close()

    # 生成样本图
    model.eval()
    K = model.K
    latent_dim = model.vae.latent_dim
    samples_per_state = 10

    fig, axes = plt.subplots(K, samples_per_state, figsize=(samples_per_state, K * 1.5))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    with torch.no_grad():
        for k in range(K):
            x = torch.zeros(samples_per_state, K).to(device)
            x[:, k] = 1.0
            z = torch.randn(samples_per_state, latent_dim).to(device)
            gen_imgs = model.vae.decode(z, x).cpu()

            true_state_idx = perm[k]
            for i in range(samples_per_state):
                if K == 1:
                    ax = axes[i]
                else:
                    ax = axes[k, i]
                ax.imshow(gen_imgs[i].view(28, 28), cmap='gray')
                ax.axis('off')
                if i == 0:
                    ax.text(-5, 14, f"S{k}\n(T{true_state_idx})",
                            fontsize=10, va='center', ha='right')
    plt.savefig("conv_final_samples.png")
    plt.close()
    print("Saved visualizations.")

def diagnose_log_bk(log_bk):
    """诊断状态区分度"""
    sorted_bk, _ = torch.sort(log_bk, dim=2, descending=True)
    gap = sorted_bk[:, :, 0] - sorted_bk[:, :, 1]
    return gap.mean().item()

# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
def train_experiment():
    DATA_FILE = "hmm_mnist_data.pt"
    if not os.path.exists(DATA_FILE):
        print("Data file not found.")
        return

    checkpoint = torch.load(DATA_FILE)
    sequences = checkpoint['sequences'].float()
    true_A = checkpoint['true_A']
    K = true_A.shape[0]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {DEVICE}, K={K}")

    # ========================================
    # 关键超参数
    # ========================================
    BATCH_SIZE = 64
    EPOCHS = 100

    # latent_dim: 控制 z 的信息容量
    # - 太大 (如 32): z 能编码类别信息，x 被忽略
    # - 太小 (如 2): z 容量不足，重建质量差，可能退化到全黑/全白
    # - 推荐: 8-16 配合高 beta_kl
    LATENT_DIM = 8

    # beta_kl: KL 散度权重 (β-VAE 思想)
    # - 高 beta (如 10-50): 强迫 z → N(0,I)，减少 z 携带的信息
    # - 这迫使模型依赖 x 来区分类别
    BETA_KL = 20.0

    # scale_factor: log_bk 的缩放因子
    # - 控制 Gumbel softmax 的 logits 范围
    SCALE_FACTOR = 50.0

    LR = 5e-4
    # HMM_LR = 5e-4
    # VAE_LR = 5e-4
    TAU_START = 1.5
    TAU_MIN = 0.3

    # ========================================

    dataset = TensorDataset(sequences)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = HMM_VAE(K, LATENT_DIM, device=DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # # 分离参数组
    # hmm_params = [model.log_A, model.log_pi]
    # vae_params = [p for n, p in model.named_parameters() if 'log_A' not in n and 'log_pi' not in n]

    # # [修改] 给 HMM 10倍~20倍的学习率
    # optimizer = optim.Adam([
    #     {'params': vae_params, 'lr': VAE_LR},       # VAE 保持慢稳
    #     {'params': hmm_params, 'lr': HMM_LR}        # HMM 加速冲刺
    # ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    print(f"Config: latent_dim={LATENT_DIM}, beta_kl={BETA_KL}, scale={SCALE_FACTOR}")
    print("="*60)
    print("Start Training...")

    best_err = float('inf')

    for epoch in range(EPOCHS):
        # Temperature annealing
        tau = max(TAU_MIN, TAU_START * (0.97 ** epoch))

        total_loss = 0
        total_emit = 0
        total_trans = 0
        total_kl = 0
        all_gaps = []

        model.train()
        for (y_batch,) in loader:
            y_batch = y_batch.to(DEVICE)
            B, T, _, _, _ = y_batch.size()
            optimizer.zero_grad()

            # 1. Compute emissions
            log_bk, z, mu, logvar = model.compute_log_bk(y_batch, scale_factor=SCALE_FACTOR)

            with torch.no_grad():
                all_gaps.append(diagnose_log_bk(log_bk))

            # 2. Forward pass
            log_alpha, log_A = model.forward_pass(log_bk)

            # 3. Backward sampling with Gumbel softmax
            X = model.backward_sample_gumbel(log_alpha, log_A, tau=tau)

            # 4. Compute losses
            # Emission loss: E[sum_k X_k * log p(y|z,x=k)]
            emission_recon = -torch.sum(X * log_bk) / (B * T)

            # KL divergence: 高 beta 强迫 z → N(0,I)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (B * T)

            loss_emission = emission_recon + BETA_KL * kld

            # Transition loss
            pi, A = model.get_hmm_params()
            log_pi_safe = torch.log(pi + 1e-9)
            log_A_safe = torch.log(A + 1e-9)

            loss_init = -torch.sum(X[:, 0, :] * log_pi_safe) / B
            x_prev = X[:, :-1, :]
            x_curr = X[:, 1:, :]
            trans_score = torch.einsum('btj,btk,jk->', x_prev, x_curr, log_A_safe)
            loss_trans = -trans_score / (B * (T - 1))
            loss_transition = loss_init + loss_trans

            # Total loss (Manuscript ELBO)
            loss = loss_emission + loss_transition

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            total_emit += emission_recon.item()
            total_trans += loss_transition.item()
            total_kl += kld.item()

        scheduler.step()

        avg_loss = total_loss / len(loader)
        avg_emit = total_emit / len(loader)
        avg_trans = total_trans / len(loader)
        avg_kl = total_kl / len(loader)
        avg_gap = np.mean(all_gaps)

        # Evaluate
        with torch.no_grad():
            est_A = model.get_hmm_params()[1].cpu().numpy()
            err, _ = evaluate_alignment_hungarian(est_A, true_A)
            if err < best_err:
                best_err = err
                # 保存最佳模型
                torch.save(model.state_dict(), "best_model.pt")

        print(f"Ep {epoch:03d} | Loss {avg_loss:.2f} | Emit {avg_emit:.2f} | "
              f"KL {avg_kl:.2f} | Trans {avg_trans:.3f} | "
              f"A_err {err:.4f} (best {best_err:.4f}) | tau {tau:.2f} | gap {avg_gap:.2f}")

        if epoch % 20 == 0:
            visualize_decoder_control(model, DEVICE, epoch)

            # 额外诊断：检查 z 的分布
            with torch.no_grad():
                sample_batch = next(iter(loader))[0].to(DEVICE)
                _, _, mu_sample, logvar_sample = model.compute_log_bk(sample_batch)
                z_std = torch.exp(0.5 * logvar_sample).mean().item()
                z_mu_norm = mu_sample.norm(dim=1).mean().item()
                print(f"  [Diag] z_std={z_std:.3f}, |mu|={z_mu_norm:.3f} "
                      f"(z_std→1, |mu|→0 when beta_kl is effective)")

    # 加载最佳模型并可视化
    model.load_state_dict(torch.load("best_model.pt"))
    visualize_final_results(model, true_A, DEVICE)
    print(f"\nTraining complete. Best A_err: {best_err:.4f}")

if __name__ == "__main__":
    train_experiment()
