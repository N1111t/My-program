from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from config import *
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_data(test_size=0.2):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    split_dataset = dataset.train_test_split(test_size=test_size)

    def format_func(examples):
        texts = []
        labels = []
        for inp, out in zip(examples["input"], examples["output"]):
            text = f"<think>user\n{inp}</tool_call>\nassistant\n{out}</tool_response>"
            texts.append(text)
            labels.append(out)
        return {"text": texts, "labels": labels}  # 返回 labels

    def tokenize_func(examples):
        model_inputs = tokenizer(
            examples["text"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["labels"],
                max_length=MAX_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs

    processed_datasets = split_dataset.map(
        format_func,
        batched=True,
        remove_columns=dataset.column_names
    )

    tokenized_datasets = processed_datasets.map(
        tokenize_func,
        batched=True,
        remove_columns=["text", "labels"]  # 仅移除原始列
    ).with_format("torch")

    output_dir = os.path.join(OUTPUT_DIR, "processed_data")
    os.makedirs(output_dir, exist_ok=True)
    tokenized_datasets.save_to_disk(output_dir)

    print("数据集字段:", tokenized_datasets["train"].column_names)
    return tokenized_datasets

if __name__ == "__main__":
    output_dir = os.path.join(OUTPUT_DIR, "processed_data")
    os.makedirs(output_dir, exist_ok=True)

    # 预处理数据
    tokenized_datasets = preprocess_data(test_size=0.1)

    # 保存处理后的数据集
    tokenized_datasets.save_to_disk(output_dir)

    # 验证数据
    logger.info(f"训练集样本示例: {tokenized_datasets['train'][0]}")
    logger.info(f"测试集样本示例: {tokenized_datasets['test'][0]}")
