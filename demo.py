# !pip3 install -q -U transformers
# !pip3 install -q -U datasets
# !pip3 install -q -U bitsandbytes
# !pip3 install -q -U peft
# !pip3 install -q -U trl
# !pip3 install -q -U accelerate

import argparse
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, TrainingArguments
import transformers
import torch

from peft import LoraConfig, PeftModel
from trl import SFTTrainer

CONST_STR_PROGRAM_NAME = "챗봇"
CONST_STR_GUIDE_MODEL = "google/gemma-2-2b-it"

CONST_PATH_MODEL = "./model"

CONST_LIST_CATEGORIES = ['가정폭력', '가출경험 및 가출중 정황', '걱정', '교사', '기타 보호자', '미래/진로', '방임', '분노/짜증', '성학대', '수면', '신체손상', '신체학대', '아버지', '어머니', '자해/자살', '정서학대', '즐거움', '친구', '통증', '트라우마', '학교폭력', '행복', '형제자매']


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

class Chatbot():
    def __init__(self, model_id, quantization_config=bnb_config):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_size = "right"

    def add_sepcial_tokens(self, special_tokens:list=[]):
        for t in special_tokens:
            assert t[0] == "<" and t[-1] == ">"
        
        # 커스텀 토큰 정의
        special_tokens = {'additional_special_tokens': ['<BeginningOfCouncel>', '<NextTheme>']}

        # 토크나이저에 커스텀 토큰 추가 및 모델 임베딩 크기 조정
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # 모델 설정 업데이트
        self.model.config.update(special_tokens)

    def ask_answer(self, chat, friendly:bool=False):
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
        if friendly:
            outputs = self.model.generate(input_ids=inputs.to(self.model.device), max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.2, do_sample=True)
        else:
            outputs = self.model.generate(input_ids=inputs.to(self.model.device), max_new_tokens=150)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("model\n")[-1]

def print_log(msg:str):
    print(f"[{datetime.now().strftime('%Y.%m.%d - %H:%M:%S')}] {msg}")

def parsing_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', '--model', metavar='N', type=str, default="gemma-2-2b-it-eym-ko",
                        help='counseling model path')
    parser.add_argument("--debug", action="store_true",
                        help="print debugging messages")

    return parser.parse_args()

def init(args):
    if args.debug:
        print("============================================")
        print_log("Get argparse:")
        print_log(f"model info: {args.model}")
        print_log(f"debug mode: {args.debug}")
        print("============================================")
    
    print(f"{CONST_STR_PROGRAM_NAME}을 초대하고 있습니다. 잠시만 기다려주세요.")
    
def get_category(answer):
    for category in CONST_LIST_CATEGORIES:
        if category in answer:
            return category
    
    return None

def main(args):
    init(args)

    guide_bot = Chatbot(CONST_STR_GUIDE_MODEL)
    if args.debug:
        print_log(f"Success to load model: {CONST_STR_GUIDE_MODEL}")
    counsel_bot = Chatbot(args.model)
    if args.debug:
        print_log(f"Success to load model: {args.model}")

    print(f"{CONST_STR_PROGRAM_NAME}이 나타났어요.")
    # # v1.0
    # print("고민을 입력하세요: ")
    # user_worry = input()
    # if args.debug:
    #     print_log(f"사용자 입력 내용: {user_worry}")
    # prompt_worry = f"""다음 고민을 보고, 주어진 고민 데이터셋에서 하나의 고민으로 분류하시오. 출력 형식은 "분류 결과: " 입니다.\n고민:"{user_worry}"\n고민 데이터셋:{CONST_LIST_CATEGORIES}"""
    # if args.debug:
    #     print_log(f"Ask category to guide bot: {prompt_worry}")
    # chat_worry = [{"role":"user", "content":f"{prompt_worry}"}]
    # answer = guide_bot.ask_answer(chat_worry)
    # if args.debug:
    #     print_log(f"Answer category from guide bot : {chat_worry}")

    # category = get_category(answer)
    # if category == None:
    #     if args.debug:
    #             print_log(f"Fail to get category, user input: {user_worry}")
    #     raise ValueError(f"Can't get category from {CONST_LIST_CATEGORIES}")
    # if args.debug:
    #         print_log(f"Success to get category: {category}")

    # print("\n" + answer)

    # v2.0
    print("다음 주제를 보고, 상담 받고 싶은 주제를 선택하세요.")
    print(f"주제: {CONST_LIST_CATEGORIES}")
    print(f"예시: {CONST_LIST_CATEGORIES[2]}")
    
    user_worry = input()

    while (user_worry not in CONST_LIST_CATEGORIES):
        print("알맞는 주제를 찾을 수 없어요. 다음 주제에서 선택해주세요.")
        print(f"{CONST_LIST_CATEGORIES}")
    
    category = user_worry
    
    chat = []
    
    new_chat = category
    while new_chat != "///":
        chat.append({"role":"user", "content":f"{new_chat}"})
        
        answer = counsel_bot.ask_answer(chat)

        if "수고하셨습니다." in answer:
            # # v1.0
            # chat.append({"role":"assistant", "content":f"{answer}"})
            # end_of_chat = "위 상담 내용을 바탕으로 나에게 공감해주는 말을 해줘. 출력 형식은 [상담 종료: ...]을 사용해"
            # chat.append({"role":"user", "content":f"{end_of_chat}"})
            # answer = guide_bot.ask_answer(chat, friendly=True)
            # answer = answer.split("]")[0].split("[")[1]
            # print(f"\n{CONST_STR_PROGRAM_NAME}: \n", answer, '\n')
            
            # # v2.0
            # end_of_chat = {"role": "user", "content": f"다음 두 사람의 대화를 듣고 마지막 말에 대한 공감의 대화를 추가해줘. 주제는 {category}이고 대화 내용은 다음과 같아. 대화 내용: {[x['content'] for x in chat]}. 형식은 '공감 내용:'하고 공감 내용을 적어줘"}
            # answer = guide_bot.ask_answer(end_of_chat, friendly=True)
            # print(f"\n{CONST_STR_PROGRAM_NAME}: \n", answer, '\n')

            # v3.0
            print(f"\n{CONST_STR_PROGRAM_NAME}: \n", "그렇구나. 너에 대해서 알아가서 너무 좋아. 다음에도 이야기 하자.", '\n')
            
            return
        else:
            print(f"\n{CONST_STR_PROGRAM_NAME}: \n", answer, '\n')
            
        chat.append({"role":"assistant", "content":f"{answer}"})

        print("user: ")
        new_chat = input()
        if args.debug:
            print_log(f"사용자 입력 내용: {new_chat}")

    return

if __name__ == "__main__":
    args = parsing_args()
    main(args)
