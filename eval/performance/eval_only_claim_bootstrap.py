import json
import os
import glob
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

# 直接python 输出到当前dir
  
# 配置日志
logging.basicConfig(
    filename="test-cv-rawfc-bootstrap.txt",
    filemode="w",          
    format="%(message)s",  
    level=logging.INFO,
    encoding="utf-8"   
)
log = logging.getLogger(__name__)

def bootstrap_metric(y_true, y_pred, classes, n_boot=10000, metric='f1_macro', seed=42):
    """
    y_true, y_pred: list
    classes: list of valid labels
    metric: 'f1_macro', 'accuracy', 'precision', 'recall'
    返回: mean, lower95, upper95
    """
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n) 
        yt_sample = y_true[idx]
        yp_sample = y_pred[idx]
        
        if metric == 'f1_macro':
            val = f1_score(yt_sample, yp_sample, labels=classes, average='macro')
        elif metric == 'accuracy':
            val = accuracy_score(yt_sample, yp_sample)
        elif metric == 'precision':
            val = precision_score(yt_sample, yp_sample, labels=classes, average='macro')
        elif metric == 'recall':
            val = recall_score(yt_sample, yp_sample, labels=classes, average='macro')
        else:
            raise ValueError(f"Unknown metric {metric}")
        stats.append(val)
    
    mean = np.mean(stats)
    lower = np.percentile(stats, 2.5)
    upper = np.percentile(stats, 97.5)
    return mean, lower, upper

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
        predicted = predicted_dict[base_id].split(':')[-1].strip()
        
        if base_id in predicted_dict:
            
            for turn in item ['conversations']:
                if turn['from'] == 'gpt':
                    label = turn['value'].split(':')[-1].strip()
            combined = {
                'id': item['id'],
                'question_id': base_id,
                'label': label,
                'predicted': predicted,
            }
            result.append(combined)
    
    return result


def get_performance(FAST_CHAT_FORMAT, TEST_SET_PATH, BENCH_NAME, MODEL_NAME):
    res = combine(FAST_CHAT_FORMAT, TEST_SET_PATH)
    log.info('检查长度:\n%s', len(res))
    log.info('--'*48)

    y_true = [sample['label'] for sample in res]
    y_pred = [sample['predicted'] for sample in res]
    ids = [sample['id'] for sample in res]  # 取出id

    if "liar-raw" in BENCH_NAME:
        classes = ['TRUE', 'FALSE', 'HALF-TRUE']  
    elif "raw-fc" in BENCH_NAME:
        classes = ['true', 'false', 'half'] 
    elif "averitec" in BENCH_NAME:
        classes = ['Supported', 'Refuted', 'Not Enough Evidence']

    valid_classes = set(classes)

    filtered = []
    dropped = []  

    for id_, yt, yp in zip(ids, y_true, y_pred):
        if yt in valid_classes and yp in valid_classes:
            filtered.append((yt, yp))
        else:
            dropped.append((id_, yt, yp))

    if filtered:
        y_true_filtered, y_pred_filtered = zip(*filtered)
    else:
        y_true_filtered, y_pred_filtered = [], []

    log.info("⚠️ 被过滤掉的非法样本：")
    for id_, yt, yp in dropped[:2]:
        log.info(f"id: {id_}, y_true: '{yt}', y_pred: '{yp}'")

    if len(dropped) == 0:
        cnt = dict()
        for exp in res:
            cnt[exp['predicted']] =  cnt.get(exp['predicted'], 0) + 1 
        log.info(f"检查标签分布:\n{cnt}")

    log.info("⚠️ 被过滤掉的非法样本数量：")
    log.info(len(dropped))

    log.info('--'*48)
    log.info('总表现')

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=classes, average='macro')
    precision_weighted = precision_score(y_true, y_pred, labels=classes, average='macro')
    recall_weighted = recall_score(y_true, y_pred, labels=classes, average='macro')

    log.info(f"Accuracy: {accuracy:.4f}")
    log.info(f"\nMacro-F1: {macro_f1:.4f}")
    log.info(f"\nMacro-Precision: {precision_weighted:.4f}")
    log.info(f"\nMacro-Recall: {recall_weighted:.4f}")


    log.info('--'*48)
    log.info('分表现')

    metrics = ['precision', 'recall', 'f1_macro', 'accuracy']
    metric_names = {'precision':'P', 'recall':'R', 'f1_macro':'F', 'accuracy':'ACC'}

    log.info("⚡ Bootstrap 95%% CI:")
    for m in metrics:
        mean, lower, upper = bootstrap_metric(y_true_filtered, y_pred_filtered, classes, metric=m)
        log.info(f"{metric_names[m]}: {mean:.4f} [{lower:.4f}, {upper:.4f}]")

if __name__ == "__main__":
    # BENCH_NAME = "averitec-cev-dev-ordinal" 
    # BENCH_NAME = "averitec-cev-train-ordinal" 
    # BENCH_NAME = "liar-raw-cv-test" 
    # BENCH_NAME = "liar-raw-cv-train" 
    # BENCH_NAME = "liar-raw-cev-test-ordinal" 
    # BENCH_NAME = "liar-raw-cev-train-ordinal" 
    BENCH_NAME = "raw-fc-cv-test"
    # BENCH_NAME = "raw-fc-cv-train"
    # BENCH_NAME = "raw-fc-cev-test"   
    # BENCH_NAME = "raw-fc-cev-train-ordinal"   
    # BENCH_NAME = "raw-fc-cev-test-ordinal"   

    TEST_SET_PATH = "/home/comp/24483737/fc/data/rawfc-post/small-original-half/test_rawfc_cv.json"
    # TEST_SET_PATH = "/home/comp/24483737/fc/data/rawfc-post/small-original-half/train_rawfc_cv.json"
    # TEST_SET_PATH = "/home/comp/24483737/fc/data/liar-raw-post/not_ordinal/test_liar_raw_cv.json"
    # TEST_SET_PATH = "/home/comp/24483737/fc/data/liar-raw-post/not_ordinal/train_liar_raw_cv.json"
    # ordinal也一样label
    # TEST_SET_PATH = "/home/comp/24483737/fc/data/post-averitec/lalast/dev_averitec_cev_remove.json"
    # TEST_SET_PATH = "/home/comp/24483737/fc/data/post-averitec/last/train_averitec_cev.json"

    base_dir = f"/home/comp/24483737/FastChat/fastchat/llm_judge/data/{BENCH_NAME}/model_answer"

    pattern = f"{base_dir}/*.jsonl"
    file_list = glob.glob(pattern)

    for f in file_list:
        sample = f.split('/')
        MODEL_NAME = sample[-1].rstrip('./jsonl')
        FAST_CHAT_FORMAT = f"/home/comp/24483737/FastChat/fastchat/llm_judge/data/{BENCH_NAME}/model_answer/{MODEL_NAME}.jsonl"
        log.info('**'*48)
        log.info("%s GO!!!!!!!!!!!", MODEL_NAME)
        try:
            get_performance(FAST_CHAT_FORMAT, TEST_SET_PATH, BENCH_NAME, MODEL_NAME)
        except:
            continue
