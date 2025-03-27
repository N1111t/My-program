from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    pipeline, EarlyStoppingCallback
)
from config import *
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model():
    # ==== 1. 加载 Tokenizer ====
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ==== 2. 数据准备 ====
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt"
    )

    processed_dataset = load_from_disk(os.path.join(OUTPUT_DIR, "processed_data"))
    train_dataset = processed_dataset["train"]
    eval_dataset = processed_dataset["test"]

    # ==== 3. 模型配置 ====
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["lm_head"]
    )

    # 移除Flash Attention配置
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        # 移除以下行
        # attn_implementation="flash_attention_2"
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True  # 保留梯度检查点
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ==== 4. 训练参数 ====
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        label_smoothing_factor=0.1
    )

    # ==== 5. 构建Trainer ====
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            # 需要从transformers导入EarlyStoppingCallback
            EarlyStoppingCallback(early_stopping_patience=3)
        ]
    )

    # ==== 6. 启动训练 ====
    trainer.train()

    # ==== 7. 模型保存 ====
    model.eval()
    model = model.merge_and_unload()

    output_path = os.path.join(OUTPUT_DIR, "final_model")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # ==== 8. 生成测试 ====
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )

    test_samples = eval_dataset.select(range(2))
    for sample in test_samples:
        input_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        true_output = tokenizer.decode(sample["labels"], skip_special_tokens=True)

        outputs = generator(input_text)
        generated_text = outputs[0]["generated_text"][len(input_text):]

        logger.info(f"\n[输入]: {input_text}")
        logger.info(f"[真实输出]: {true_output}")
        logger.info(f"[生成结果]: {generated_text}")


if __name__ == "__main__":
    train_model()
