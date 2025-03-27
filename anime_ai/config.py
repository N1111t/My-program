import os
import random
import numpy as np
import torch

# 基础配置优化
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 避免分词器警告

# 随机种子（新增）
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 硬件配置优化
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2  # 建议保持动态调整：
# 可添加以下逻辑自动适配显存：
try:
    BATCH_SIZE = 4
    # 进行显存预估计算
except RuntimeError:
    torch.cuda.empty_cache()
    BATCH_SIZE = 2

# 混合精度训练（新增）
FP16 = True  # 需要配合相应的训练框架配置

# 模型配置优化
MODEL_NAME = "DeepSeek-R1-Distill-Qwen-1.5B"
LORA_CONFIG = {
    "r": 32,          # 建议保持动态调整
    "lora_alpha": 64, # 新增参数（通常设为r的2倍）
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    "lora_dropout": 0.1,
    "bias": "none"
}

# 路径配置优化
DATA_PATH = "./data/dataset.json"
OUTPUT_DIR = "./anime_ai_output"

# 自动创建输出目录（新增）
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据处理配置（新增建议）
MAX_LENGTH = 512   # 根据实际数据分布调整
CUTOFF_LEN = 256   # 文本截断长度
