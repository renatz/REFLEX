import json
import requests
import urllib3
import openai
import logging

log_file = './prompt_error.log'
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class PostRobot:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.history_message = []

    def request_chatgpt(self, parameters):
        # url = "https://api.oneabc.org/v1/chat/completions"
        url = "https://api.shubiaobiao.cn/v1/chat/completions"
        # key = 'sk-MURxdBOjdMXAZnoXC48052Df57C04095B4Ac293eE2B51d4c'
        key = 'sk-RD8QagYIVzuMOQF1E4Be9c4aF4574aF598C8F41225E47f3e'

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        
        raw_response = requests.post(url, headers=headers, json=parameters, verify=False)
        response = json.loads(raw_response.content.decode("utf-8"))
        
        try:
            content = response["choices"][0]["message"]["content"]
            flag = True
        except:
            content = response["error"]["code"]
            flag = False
        return flag, content
    
    # def request_chatgpt(self,  parameters):
    #     openai.api_key = "sk-2f6VkychzhV5aUWl7aF6921886194482A0388d5e8c0157Bc"  
    #     openai.api_base = "https://api.ai-gaochao.cn/v1"

    #     # 判定，开温度为0
    #     try:
    #         completion = openai.ChatCompletion.create(
    #             model=parameters['model'],
    #             messages=parameters['messages'],
    #             temperature=0  
    #         )
            
    #         msg = None
    #         choices = completion.get('choices', None)
    #         if choices:
    #             msg = choices[0]['message']['content']
    #         else:
    #             msg = completion['message']['content']
    #         return (True, msg)
    #     except Exception as err:
    #         if 'Detected an error in the prompt' in f'{err}':
    #             logger.info(f'{err}\n')
    #             logger.info('Messages: %s', json.dumps(parameters['messages']))
    #         return (False, f'OpenAI API 异常: {err}')

    def generate(self, api_key, new_message, role=None, args=None, use_vpn=True):
        if role is not None:
            role = {
                "role": "system",
                "content": role,
            }
        if len(self.history_message) == 0 and role is not None:
            self.history_message.append(role)
        temp_message = self.history_message.copy()
        temp_message.append({"role": "user", "content": new_message})
        parameters = {
            "model": self.model_name,
            "messages": temp_message
        }
        if args is not None:
            for key in args.keys():
                parameters[key] = args[key]
        # flag, response = self.request_chatgpt(api_key, org_id, parameters, use_vpn)
        # flag, response = self.request_chatgpt(api_key, parameters, use_vpn)
        flag, response = self.request_chatgpt(parameters)
        
        if flag == True:
            self.history_message.append({"role": "user", "content": new_message})
            self.history_message.append({"role": "assistant", "content": response})
        return flag, response
