import json
import re
import pandas as pd
import os
import glob
import logging
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    filename="log/log_quality_base_raw.txt",
    # filename="log_quality_steer.txt",
    filemode="w",
    format="%(message)s",
    level=logging.INFO,
    encoding="utf-8"
)
log = logging.getLogger(__name__)

def clean_score(score_str: str):
    if not isinstance(score_str, str):
        return score_str

    match = re.search(r'\{[\s\S]*?\}', score_str)
    if not match:
        print("❌ 找不到 JSON object")
        return None

    s = match.group()

    s = re.sub(
        r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:',
        r'\1"\2":',
        s
    )

    s = s.replace("'", '"')

    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        print("❌ JSON 解析失败")
        print("最终 JSON:", s)
        print("错误:", e)
        return None


if __name__ == "__main__":
    input_dirs = [
        "/home/comp/24483737/fc/eval/res/all-base-raw",
        # "/home/comp/24483737/fc/eval/res/all-steer",
    ]

   
    file_df_pool = defaultdict(list)

    for input_dir in input_dirs:
        json_files = glob.glob(os.path.join(input_dir, "*.json"))

        for file_path in json_files:
            data_list = []

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    if "score" in item:
                        item["score"] = clean_score(item["score"])
                    data_list.append(item)

            scores_list = []
            for sample in data_list:
                scores_list.append(sample["score"])

            df = pd.DataFrame(scores_list)

            mean_scores = df.mean()

            filename = os.path.basename(file_path)

            log.info(f"Dir: {input_dir}")
            log.info(f"File: {filename}")
            log.info("Mean:")
            log.info(mean_scores)
            log.info("**" * 48)

            file_df_pool[filename].append(df)

    log.info("\n\n========== Cross-Dir Statistics ==========\n")

    for filename, df_list in file_df_pool.items():

        if len(df_list) != len(input_dirs):
            log.info(f"⚠️ Warning: {filename} has {len(df_list)} dirs only")

        merged_df = pd.concat(df_list, axis=0, ignore_index=True)

        mean_scores = merged_df.mean()
        std_scores = merged_df.std()

        log.info(f"File: {filename}")
        log.info("Overall Mean:")
        log.info(mean_scores)
        log.info("Overall Std:")
        log.info(std_scores)

        log.info("Mean ± Std:")
        for k in mean_scores.index:
            log.info(f"{k}: {mean_scores[k]:.4f} ± {std_scores[k]:.4f}")

        log.info("##" * 48)
