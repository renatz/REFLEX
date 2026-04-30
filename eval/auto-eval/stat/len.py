import os
import json
from transformers import LlamaTokenizer
import logging
import sys
sys.stdout.reconfigure(encoding='utf-8')


# 搜invalid_prompt
# 配置日志
logging.basicConfig(
    filename="log/token_statistics.txt",
    filemode="w",         
    format="%(message)s",  
    level=logging.INFO,
    encoding="utf-8"   
)
log = logging.getLogger(__name__)

def compute_average_explanation_tokens_per_file(directory):

    tokenizer = LlamaTokenizer.from_pretrained("/home/comp/24483737/model/llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9")

    for filename in os.listdir(directory):
        if filename.endswith(".json") or filename.endswith(".jsonl"):
            file_path = os.path.join(directory, filename)
            token_counts = []

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        log.info(f"Skipping invalid JSON line in {file_path}")
                        continue

                    predicted_text = data.get("predicted", "")
                    if "Explanation:" in predicted_text:
                        explanation = predicted_text.split("Explanation:", 1)[1].strip()
                    else:
                        explanation = ""

                    tokens = tokenizer.tokenize(explanation)
                    token_counts.append(len(tokens))

            if token_counts:
                avg_tokens = sum(token_counts) / len(token_counts)
                log.info(f"{file_path}: {avg_tokens:.2f} tokens per sample")
            else:
                log.info(f"{file_path}: No explanations found")

if __name__ == "__main__":
    data_dir = "/home/comp/24483737/fc/eval/all-steer-merged"
    # data_dir = "/home/comp/24483737/fc/eval/all-base-raw-merged"
    compute_average_explanation_tokens_per_file(data_dir)
