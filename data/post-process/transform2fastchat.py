import json

def read_json_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        json_data = json.load(file)
    return json_data

def transform(input_path, output_path):
    before = read_json_file(input_path)
    for ele in before:
        line_dict = {}
        line_dict['question_id'] = ele['id']
        line_dict['category'] = 'abbr'
        
        q = ele['conversations'][0]['value']

        line_dict['turns'] = [q]

        with open(output_path, 'a', encoding='utf-8') as f:
            json.dump(line_dict, f, ensure_ascii=False)
            f.write('\n')
            
# transform(
#           "/home/comp/24483737/fc/data/rawfc-post/small-original-half/train_rawfc_cev.json",
#           "/home/comp/24483737/fc/data/rawfc-post/small-original-half/train_rawfc_cev_fastchat.json",
#           )

            
# transform(
#           "/home/comp/24483737/fc/data/train_liar_raw_cev.json",
#           "/home/comp/24483737/fc/data/train_liar_raw_cev_fastchat.json",
#           )

# transform(
#           "/home/comp/24483737/fc/data/rawfc-post/small-original-half/train_rawfc_cev_ordinal.json",
#           "/home/comp/24483737/fc/data/rawfc-post/small-original-half/train_rawfc_cev_ordinal_fastchat.json",
#           )

# transform(
#           "/home/comp/24483737/fc/data/rawfc-post/small-original-half/test_rawfc_cev_ordinal.json",
#           "/home/comp/24483737/fc/data/rawfc-post/small-original-half/test_rawfc_cev_ordinal_fastchat.json",
#           )

# transform(
#           "/home/comp/24483737/fc/data/post-averitec/dev_averitec_cev.json",
#           "/home/comp/24483737/fc/data/post-averitec/dev_averitec_cev_fastchat.json",
#           )

transform(
          "/home/comp/24483737/fc/data/post-averitec/last/train_averitec_cev.json",
          "/home/comp/24483737/fc/data/post-averitec/last/train_averitec_cev_fastchat.json",
          )