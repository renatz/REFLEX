import logging
import json
import time
import json
import re
from tqdm import tqdm
from multiprocessing import Pool, Lock

import os
import glob
from gpt import GPT
    
import sys
sys.stdout.reconfigure(encoding='utf-8')

BATCH_SIZE = 17
NUM_PROCESS = 30
PROMPT_PATH = f'./prompt.txt'


unique_ids = set()
unique_lock = Lock()

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

def read_txt_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        content = file.read()
    return content

def generate_eval_list(prompt_path, data_path):
    to_eval_list = [] 
    data_list = read_single_line(data_path)
    prompt_str = read_txt_file(prompt_path)

    for line_dict in data_list:
        to_eval_dict = {}

        question_id = line_dict['question_id']
        prediction = line_dict['predicted']
        claim = line_dict['claim']
        label = line_dict['label']

        new_content_str = prompt_str.format(
            claim = claim,
            label = label,
            verdict = prediction
        )
        
        
        to_eval_dict['id'] = question_id
        to_eval_dict['conversations'] = new_content_str
        to_eval_list.append(to_eval_dict)
    return to_eval_list

def send_request(session_id:int, req:str):  
    gpt = GPT(user_name="kongchuyi", model_name="gpt-3.5-turbo-0613-16k", use_vpn=False)

    rep = None
    err = 1
    while err or (not rep) or (rep == 'APIKey Error') or ('invalid_api_key' in rep) or ('account_deactivated' in rep) or ('rate_limit_exceeded' in rep) or ('insufficient_quota' in rep) or ('获取请求参数user_name和parameters失败' in rep) or ('请求官方API失败。Error code:200') in rep or ('OpenAI API 异常') in rep or ('bad_response_status_code' in rep):
        try:
            _, rep = gpt.call(req)

            if rep:
                if 'context_length_exceeded' in rep: 
                    return None
            err = 0
        except BaseException as e:  
            print(f'\nThe {session_id}-session: \n {type(e)} \n {e}')
    return rep
    
def write_to_file(results, write_into_path):
    with open(write_into_path, 'a', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)
        f.write('\n')

def process_batch(batch:list, output_path:str):
    for session_dict in tqdm(batch):
        
        if session_dict['id'] not in unique_ids:
            with unique_lock:
                unique_ids.add(session_dict['id'])
                
            result = send_request(session_dict['id'], session_dict['conversations']) 
            
            
            if result:
                result_dict = {
                        'id': session_dict['id'],
                        'prompt': session_dict['conversations'],
                        'score': result
                }
                write_to_file(result_dict, output_path)

if __name__ == '__main__':
    start_time = time.time()
    data_path = '/home/comp/24483737/fc/eval/all-base-merged/llama-2-rawfc-cvi-remove_vox-sep-ep2-utf8.json'

    input_dir = "/home/comp/24483737/fc/eval/all-base-raw-merged/"
    output_dir = 'all-base-raw-6'
    

    json_files = glob.glob(os.path.join(input_dir,"*.json"))
    for data_path in json_files:
        output_name = data_path.split('/')[-1]
        os.makedirs(f'/home/comp/24483737/fc/eval/res/{output_dir}/', exist_ok=True)
        output_path = f'/home/comp/24483737/fc/eval/res/{output_dir}/{output_name}'
        
        to_eval_list = generate_eval_list(PROMPT_PATH, data_path)
        
        with Pool(processes=NUM_PROCESS) as pool:
            for i in range(0, len(to_eval_list), BATCH_SIZE):
                batch_list = to_eval_list[i:i+BATCH_SIZE]
                pool.apply_async(process_batch, (batch_list, output_path ))
            pool.close()
            pool.join()

        end_time = time.time()
        print(f'Elapsted {end_time-start_time} seconds.')
