import torch
import numpy as np
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt

class HMM_MNIST_Generator:
    """
    基于 MNIST 数据集生成符合 HMM 隐状态转移规律的图片序列。
    """
    def __init__(self, digits=[0, 1, 2], data_dir='./data'):
        """
        Args:
            digits (list): 选用的数字集合，对应 HMM 的隐状态。
                           例如 [0, 1, 2] 表示状态 0->数字0, 状态 1->数字1...
            data_dir (str): MNIST 数据下载/存储路径
        """
        self.digits = digits
        self.K = len(digits)
        self.data_dir = data_dir
        
        # 1. 加载 MNIST 数据
        print(f"Loading MNIST data for digits: {digits}...")
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # 下载训练集,如果不存在则下载
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        
        # 2. 按类别整理数据，方便后续采样
        # self.data_by_class[k] 存储所有属于第 k 个隐状态（对应真实数字 digits[k]）的图片
        self.data_by_class = {k: [] for k in range(self.K)}
        
        count = 0
        for img, label in train_dataset:
            label = int(label)
            if label in digits:
                # 找到该 label 在我们定义的隐状态中的索引
                state_idx = digits.index(label)
                self.data_by_class[state_idx].append(img)
                count += 1
        
        print(f"Data loaded. Total valid images: {count}")
        # 将列表转换为 Tensor 以加速索引
        for k in range(self.K):
            self.data_by_class[k] = torch.stack(self.data_by_class[k])
            print(f"State {k} (Digit {digits[k]}): {len(self.data_by_class[k])} images")

    def generate_sequences(self, num_sequences, seq_len, pi, A):
        """
        生成 HMM 观测序列 (图片) 和 隐状态序列 (标签)。
        
        Args:
            num_sequences (int): 生成多少条序列
            seq_len (int): 每条序列的长度
            pi (np.array): 初始状态概率向量 [K]
            A (np.array): 状态转移矩阵 [K, K]
            
        Returns:
            sequences (Tensor): [num_sequences, seq_len, 1, 28, 28]
            state_labels (Tensor): [num_sequences, seq_len]
        """
        sequences = []
        state_labels = []
        
        print(f"Generating {num_sequences} sequences with length {seq_len}...")
        
        for _ in range(num_sequences):
            seq_imgs = []
            seq_states = []
            
            # --- HMM 采样过程 ---
            
            # 1. 初始时刻 t=0
            # 根据初始概率 pi 选择第一个状态
            current_state = np.random.choice(self.K, p=pi)
            
            for t in range(seq_len):
                # 记录当前隐状态
                seq_states.append(current_state)
                
                # Emission (发射): 从当前状态对应的图片池中随机选一张
                # 对应论文中的 p(y|x)
                pool_size = len(self.data_by_class[current_state])
                img_idx = np.random.choice(pool_size)
                img = self.data_by_class[current_state][img_idx]
                seq_imgs.append(img)
                
                # Transition (转移): 根据转移矩阵 A 决定下一个状态
                # 对应论文中的 p(x_t | x_{t-1})
                current_state = np.random.choice(self.K, p=A[current_state])
            
            sequences.append(torch.stack(seq_imgs))
            state_labels.append(torch.tensor(seq_states, dtype=torch.long))
            
        return torch.stack(sequences), torch.stack(state_labels)

def visualize_sequence(sequence, states, save_path="sample_seq.png"):
    """
    可视化一条生成的序列，验证逻辑是否正确
    """
    seq_len = sequence.size(0)
    fig, axes = plt.subplots(1, seq_len, figsize=(seq_len * 1.5, 2))
    
    for t in range(seq_len):
        img = sequence[t].squeeze().numpy()
        state = states[t].item()
        
        axes[t].imshow(img, cmap='gray')
        axes[t].set_title(f"t={t}\nState={state}")
        axes[t].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.close()

def visualize_decoder_control(model, device, epoch):
    """
    画图验证：固定 z，改变 x，看图片是否变化。
    如果每一行的图片随着列变化而变化，说明 FiLM 生效了。
    """
    model.eval()
    K = model.K
    latent_dim = model.vae.latent_dim
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


if __name__ == "__main__":
    # # ================= 配置参数 =================
    # SAVE_FILE = "hmm_mnist_data.pt"
    # NUM_SEQS = 1000       # 训练集大小
    # SEQ_LEN = 10          # 序列长度
    # DIGITS = [0, 1, 2]    # 隐状态对应的数字
    
    # # 自定义 HMM 参数 (Ground Truth)
    # # 设定一个强规则: 0 -> 1 -> 2 -> 0 (循环)
    # # 这样我们可以通过观察生成的图片是否遵循这个顺序来验证代码
    # TRUE_PI = np.array([1.0, 0.0, 0.0]) # 总是从 0 开始
    # TRUE_A = np.array([
    #     [0.05, 0.90, 0.05], # 0 大概率变成 1
    #     [0.05, 0.05, 0.90], # 1 大概率变成 2
    #     [0.90, 0.05, 0.05]  # 2 大概率变成 0
    # ])
    # ================= 配置参数 =================
    SAVE_FILE = "hmm_mnist_10_class.pt"
    NUM_SEQS = 10000
    SEQ_LEN = 32
    DIGITS = list(range(10))  # [0, 1, ..., 9]
    K = 10

    # 构造 10x10 的循环矩阵: 0->1->2->...->9->0
    TRUE_PI = np.ones(K) / K  # 均匀初始分布

    noise = 0.01
    TRUE_A = np.full((K, K), noise / (K - 1))
    for i in range(K):
        target = (i + 1) % K
        TRUE_A[i, target] = 1.0 - noise
    TRUE_A = TRUE_A / TRUE_A.sum(axis=1, keepdims=True)

            
    # ================= 执行生成 =================
    generator = HMM_MNIST_Generator(digits=DIGITS)
    
    # 生成数据
    seqs, labels = generator.generate_sequences(NUM_SEQS, SEQ_LEN, TRUE_PI, TRUE_A)
    
    # ================= 保存数据 =================
    print(f"Saving data to {SAVE_FILE}...")
    torch.save({
        "sequences": seqs,           # [N, T, 1, 28, 28]
        "state_sequences": labels,    # [N, T] 真实标签
        "true_A": TRUE_A,            # [K, K]
        "K": K,
        # 兼容旧字段名
        "labels": labels,
        "true_pi": TRUE_PI,
        "digits": DIGITS
    }, SAVE_FILE)
    print("Done.")
    
    # ================= 验证可视化 =================
    # 画出第一条生成的序列，检查是否符合 0->1->2 的规律
    visualize_sequence(seqs[0], labels[0])