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
    SAVE_FILE = "hmm_mnist_5_class.pt"
    NUM_SEQS = 1000
    SEQ_LEN = 16          # 序列稍微长一点，让模型看清更多转移
    DIGITS = list(range(5)) # [0, 1, ..., 4]
    K = 5

    # 构造 10x10 的循环矩阵
    # 0->1, 1->2, ..., 9->0
    TRUE_PI = np.zeros(K)
    TRUE_PI[0] = 1.0 # 从 0 开始

    TRUE_A = np.zeros((K, K))
    for i in range(K):
        target = (i + 1) % K
        TRUE_A[i, target] = 0.9  # 主概率
        # 剩余概率均匀分配给噪声
        noise = 0.1 / (K - 1)
        for j in range(K):
            if j != target:
                TRUE_A[i, j] = noise

            
    # ================= 执行生成 =================
    generator = HMM_MNIST_Generator(digits=DIGITS)
    
    # 生成数据
    seqs, labels = generator.generate_sequences(NUM_SEQS, SEQ_LEN, TRUE_PI, TRUE_A)
    
    # ================= 保存数据 =================
    print(f"Saving data to {SAVE_FILE}...")
    torch.save({
        "sequences": seqs,
        "labels": labels,
        "true_pi": TRUE_PI,
        "true_A": TRUE_A,
        "digits": DIGITS
    }, SAVE_FILE)
    print("Done.")
    
    # ================= 验证可视化 =================
    # 画出第一条生成的序列，检查是否符合 0->1->2 的规律
    visualize_sequence(seqs[0], labels[0])