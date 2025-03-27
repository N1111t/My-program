import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from config import MODEL_NAME, OUTPUT_DIR
import logging

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA调用
os.environ['TORCH_USE_CUDA_DSA'] = '1'    # 启用显存调试

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_finetuned_model():
    try:
        model_path = os.path.join(OUTPUT_DIR, "final_model")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"  # 关键修改
        )
        return model
    except Exception as e:
        logger.error("模型加载失败，请检查配置")
        raise e

def create_anime_chain():
    model = load_finetuned_model()
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(OUTPUT_DIR, "final_model"),
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.5,
        pad_token_id=tokenizer.eos_token_id
    )

    anime_prompt = PromptTemplate.from_template(
        "<think>user\n{question}</tool_call>\n"
        "assistant\n"
        "好的，我现在需要处理用户的提问：{question}\n"
        "首先，我要分析用户的需求，可能涉及二次元相关的知识或角色扮演。\n"
        "接下来，按照设定的角色性格（傲娇、活泼）来构建回答。\n"
        "最后，确保回复符合口语化、简洁的要求，避免使用专业术语。"
        "</think>"
    )

    return LLMChain(llm=HuggingFacePipeline(pipeline=pipe), prompt=anime_prompt)

if __name__ == "__main__":
    chain = create_anime_chain()
    test_questions = [
        "可以帮我写作业吗？",
        "你喜欢什么动漫？",
        "能推荐一家拉面店吗？"
    ]
    for q in test_questions:
        logger.info(f"用户：{q}")
        response = chain.run(q)
        logger.info(f"AI：{response}\n{'='*30}")
