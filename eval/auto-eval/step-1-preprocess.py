import torch
import os
import json
import numpy as np
from difflib import SequenceMatcher 
import re
import glob
import sys
sys.stdout.reconfigure(encoding='utf-8')

def read_single_line(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        session_list = file.readlines()
    new_list = []
    for session_str in session_list:
        new_list.append(eval(session_str))
    return new_list


def compute_length(data):
    len_l=[]
    for sample in data:
        a = sample['choices'][0]['turns'][0] 
        len_l.append(len(a))
    print('Mean length', np.mean(len_l))

def fix_unicode_escape(text):
    return text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")

def remove_dup(text):
    text = re.sub(r'https://www\.[^\s]+', '', text)

    raw_sentences = re.split(r'(?<=[.!?\n])\s*', text.strip())

    seen = set()
    unique_sentences = []
    
    for sentence in raw_sentences:
        clean = sentence.strip()
        if not clean:
            continue
        if not re.search(r'[.!?]$', clean):
            continue
        if clean not in seen:
            seen.add(clean)
            unique_sentences.append(clean)
    
    result = ' '.join(unique_sentences)
            

    return result

def is_similar(a, b, threshold=0.6): 
    return SequenceMatcher(None, a, b).ratio() >= threshold

def remove_similar_lines(text):
    lines = re.split(r'(?<=[.!?\n])\s*', text.strip())
    unique_lines = []
    for line in lines:
        if not any(is_similar(line, existing) for existing in unique_lines):
            unique_lines.append(line)
    
    result = ' '.join(unique_lines)
    diff = len(text)-len(result)
    print('Remove_similiar_lines:', diff)
    return result

def clean(data):
    cleaned = []
    null_cnt = 0
    for sample in data:
        a = sample['choices'][0]['turns'][0] 
        fixed = fix_unicode_escape(a)
        removed = remove_dup(fixed) 
        rremoved = remove_similar_lines(removed)
      
        if len(rremoved) == 0:
            null_cnt +=1
            continue
        sample['choices'][0]['turns'][0] = rremoved
        if not sample['choices'][0]['turns'][0]:
            continue
        cleaned.append(sample)
    print(f"Null Count:{null_cnt}")
    print(f"Final Length:{len(cleaned)}")
    return cleaned



if __name__ == "__main__":
    # OUTPUT_DIR="all-base"
    # dirs = [
    #     "/home/comp/24483737/FastChat/fastchat/llm_judge/data/averitec-cev-dev-ordinal/model_answer/bl",
    #     "/home/comp/24483737/FastChat/fastchat/llm_judge/data/liar-raw-cev-test-ordinal/model_answer/bl",
    #     "/home/comp/24483737/FastChat/fastchat/llm_judge/data/liar-raw-cv-test/model_answer/bl",
    #     "/home/comp/24483737/FastChat/fastchat/llm_judge/data/raw-fc-cev-test-ordinal/model_answer/bl",
    #     "/home/comp/24483737/FastChat/fastchat/llm_judge/data/raw-fc-cv-test/model_answer/bl",
    # ]
    # jsonl_files = []
    # for d in dirs:
    #     for root, _, files in os.walk(d):
    #         for f in files:
    #             if f.endswith(".jsonl"):
    #                 jsonl_files.append(os.path.join(root, f))

    # steer
    OUTPUT_DIR="all-steer"
    dirs = [
        "/home/comp/24483737/FastChat/fastchat/llm_judge/data/averitec-cev-dev-ordinal/model_answer/cv-cvi",
        "/home/comp/24483737/FastChat/fastchat/llm_judge/data/averitec-cev-dev-ordinal/model_answer/cvi-cvi",

        "/home/comp/24483737/FastChat/fastchat/llm_judge/data/liar-raw-cv-test/model_answer/cv-cvi",
        "/home/comp/24483737/FastChat/fastchat/llm_judge/data/liar-raw-cv-test/model_answer/cvi-cvi",

        "/home/comp/24483737/FastChat/fastchat/llm_judge/data/raw-fc-cv-test/model_answer/cv-cvi",
        "/home/comp/24483737/FastChat/fastchat/llm_judge/data/raw-fc-cv-test/model_answer/cvi-cvi",
    ]
    keywords = ["cvi-cvi", "cevio-cevio", "on-cvi", "on-cevio"]
    jsonl_files = []
    for d in dirs:
        for f in glob.glob(f"{d}/*.jsonl"):
            fname = os.path.basename(f)
            if any(k in fname for k in keywords):
                jsonl_files.append(f)

    for input_file in jsonl_files:
        data= read_single_line(input_file)
        print('之前len: ', len(data))
        compute_length(data)
        cleaned_data = clean(data)

        print('之后len: ', len(cleaned_data))

        output_path = input_file.split('/')[-1].rstrip('.jsonl')
        os.makedirs(f'/home/comp/24483737/fc/eval/{OUTPUT_DIR}/', exist_ok=True)
        with open(f"/home/comp/24483737/fc/eval/{OUTPUT_DIR}/{output_path}.json", "a", encoding="utf-8") as fout:
            for sample in cleaned_data:
                json.dump(sample, fout, ensure_ascii=False)
                fout.write('\n')