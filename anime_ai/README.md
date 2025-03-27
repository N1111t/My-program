二次元风格AI助手开发项目

项目概述
这是一个用于开发具有独特回答风格（如傲娇、活泼）的AI助手的项目。通过微调预训练的因果语言模型，并结合高效训练技术和提示工程，项目旨在打造一个个性化的对话系统。当前阶段已完成模型微调、数据处理和推理模块的开发，未来计划通过构建 RAG（Retrieval-Augmented Generation）系统，整合自定义知识库，进一步提升回答的相关性和知识广度。

项目目标
-个性化风格：训练AI生成符合特定角色性格（如傲娇、活泼）的回答。
-高效训练：在资源受限环境下（如消费级 GPU）完成大模型微调。
-知识扩展：通过 RAG 系统整合自定义知识库，使AI具备领域特定知识。

项目功能
1. 数据预处理：
   - 从 JSON 文件加载对话数据，分割为训练集（90%）和测试集（10%）。
   - 使用自定义格式（如 `<think>user\n{input}</tool_call>\nassistant\n{output}</tool_response>`）处理文本。
   - 通过分词器（`AutoTokenizer`）将文本转换为模型可用的张量格式。

2. 模型训练：
   - 基于预训练模型 `DeepSeek-R1-Distill-Qwen-1.5B`（因果语言模型）进行微调。
   - 使用 LoRA（Low-Rank Adaptation）和4位量化（BitsAndBytes）优化训练效率。
   - 支持混合精度训练（FP16）和梯度检查点，降低内存需求。

3. 推理与生成：
   - 实现风格化文本生成，支持用户输入的实时响应。
   - 使用 LangChain 和 Hugging Face Pipeline 构建对话链，结合自定义提示模板控制生成风格。

4. 未来规划：
   - 开发 RAG 系统，集成检索机制和知识库。
   - 探索多模态扩展（如 CLIP），支持图像-文本交互。

技术栈
-编程语言：Python
-深度学习框架：PyTorch
-模型与工具：
  - Hugging Face Transformers（`AutoModelForCausalLM`, `AutoTokenizer`）
  - Hugging Face Datasets
  - PEFT（LoRA）
  - BitsAndBytes（4位量化）
  - LangChain（对话链）
-硬件支持：CUDA（GPU加速）
-其他库：NumPy、os、logging

安装步骤
1. 环境要求
- Python 3.8+
- CUDA 11.0+（若使用 GPU）
- 显存：建议至少 8GB（视模型规模调整）

2. 安装依赖
```bash
pip install torch torchvision
pip install transformers datasets peft bitsandbytes
pip install langchain langchain-community
pip install numpy
```

3. 下载预训练模型
项目使用 `DeepSeek-R1-Distill-Qwen-1.5B` 模型，可通过 Hugging Face Hub 下载：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("DeepSeek-R1-Distill-Qwen-1.5B")
tokenizer = AutoTokenizer.from_pretrained("DeepSeek-R1-Distill-Qwen-1.5B")
```

4. 准备数据集
- 在 `data/` 目录下放置 `dataset.json`，格式示例：
  ```json
  [
      {"input": "你喜欢什么动漫？", "output": "哼，才不会随便告诉你呢！不过《EVA》还算不错吧。"},
      {"input": "可以帮我写作业吗？", "output": "想让我帮你？哼，自己努力一点啦，我可不是你的保姆！"}
  ]
  ```

使用方法
1. 数据预处理
```bash
python preprocess_data.py
```
- 输出：处理后的数据集保存在 `anime_ai_output/processed_data/`。

### 2. 模型训练
```bash
python train_model.py
```
- 输出：检查点保存在 `anime_ai_output/checkpoints/`，最终模型保存在 `anime_ai_output/final_model/`。

3. 推理与测试
```bash
python inference.py
```
- 示例输出：
  ```
  用户：可以帮我写作业吗？
  AI：想让我帮你？哼，自己努力一点啦，我可不是你的保姆！
  ==============================
  用户：你喜欢什么动漫？
  AI：哼，才不会随便告诉你呢！不过《EVA》还算不错吧。
  ==============================
  ```

配置说明
在 `config.py` 中调整以下参数：
- `MODEL_NAME`: 预训练模型名称。
- `BATCH_SIZE`: 训练批次大小（默认 2，可根据显存调整）。
- `FP16`: 是否启用混合精度训练（默认 True）。
- `MAX_LENGTH`: 输入序列最大长度（默认 512）。
- `OUTPUT_DIR`: 输出目录（默认 `./anime_ai_output`）。

当前成果
- 成功微调模型，能生成符合预设风格的回答，不过还待优化。
- 训练效率优化：通过4位量化和LoRA，模型可在8GB显存GPU上运行。
- 推理系统可用，支持用户交互测试。

未来计划
1.RAG系统开发：
   - 构建检索模块，整合自定义知识库（如动漫相关数据）。
   - 提升回答的知识准确性和上下文相关性。
2.多模态扩展：
   - 探索CLIP等模型，增加图像理解能力。
   - 支持图像-文本交互场景（如描述图片中的动漫角色）。
3.性能优化：
   - 进一步压缩模型，适配更低端设备。
   - 提高生成速度和回答质量。