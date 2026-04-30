import os
import json
from tqdm import tqdm

def read_json_object(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        json_data = json.load(file)
    return json_data

def read_single_line(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        session_list = file.readlines()
    new_list = []
    for session_str in session_list:
        new_list.append(eval(session_str))
    return new_list

def combine(predicted_path, grouth_truth_path):
    predicted_list = read_single_line(predicted_path)
    gt_list = read_json_object(grouth_truth_path)

    predicted_dict = {item['question_id']: item['choices'][0]['turns'][0] for item in predicted_list}

    result = []
    
    for item in gt_list:
        base_id = item['id']
        predicted_text = predicted_dict[base_id]
        
        if base_id in predicted_dict:
            for turn in item ['conversations']:
                if turn['from'] == 'gpt':
                    label = turn['value'].split(':')[-1].strip()
                if turn['from'] == 'human':
                    claim = turn['value'].split("Claim:")[-1].strip()
            combined = {
                'question_id': base_id,
                'claim': claim,
                'label': label,
                'predicted': predicted_text,
            }
            result.append(combined)
    
    return result

def merge(pre_path, label_path, output_dir, output_path):
    res = combine(pre_path, label_path)
    os.makedirs(f'/home/comp/24483737/fc/eval/{output_dir}/', exist_ok=True)
    with open(f"/home/comp/24483737/fc/eval/{output_dir}/{output_path}.json", "a", encoding="utf-8") as fout:
        for sample in tqdm(res):
            json.dump(sample, fout, ensure_ascii=False)
            fout.write('\n')

if __name__ == "__main__":

    base_dir = "/home/comp/24483737/fc/eval/all-base-clean"
    output_dir="all-base-merged"

    # base_dir = "/home/comp/24483737/fc/eval/all-steer"
    # output_dir="all-steer-merged"

    jsonl_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            # if f.endswith(".json"):
            if f.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, f))

    print(jsonl_files)
    for data_path in jsonl_files:
        output_name = data_path.split('/')[-1].rstrip('.json')
        # output_name = data_path.split('/')[-1].rstrip('.jsonl')
        if 'rawfc' in output_name: 
            label_path = f"/home/comp/24483737/fc/data/rawfc-post/small-original-half/test_rawfc_cv.json"
        elif ('liar-raw' in output_name) or ('liaraw' in output_name):
            label_path = "/home/comp/24483737/fc/data/liar-raw-post/not_ordinal/test_liar_raw_cv.json"
        else:
            label_path = "/home/comp/24483737/fc/data/post-averitec/lalast/dev_averitec_cev_remove.json"
        merge(data_path, label_path, output_dir, output_name)