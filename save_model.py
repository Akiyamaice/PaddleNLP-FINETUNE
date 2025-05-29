import os
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen1.5-110B-Chat"
cache_dir = "/data/nvme1/paddle_weights"  # 替换为您希望的缓存目录路径
os.makedirs(cache_dir, exist_ok=True)  # 确保目录存在
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, dtype="float16")