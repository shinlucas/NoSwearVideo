import whisper
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertModel, BertTokenizer
from spleeter.separator import Separator
import torch
import torch.nn as nn
import time
import os
from openvoice.api import ToneColorConverter
from melo.api import TTS

# 초기화된 모델들을 담을 전역 변수들
wisper_model, replace_model, replace_tokenizer, choice_model, choice_tokenizer, tone_color_converter, separator, tts_model = (None,) * 8

# 초기화 상태를 추적하는 변수
models_initialized = False

def initialize_models():
    global models_initialized
    global wisper_model, replace_model, replace_tokenizer, choice_model, choice_tokenizer, tone_color_converter, separator, tts_model

    print("모델 초기화 시작")

    if not models_initialized:
        start_time = time.time()

        wisper_model = whisper.load_model("medium")
        print(f"Whisper model loaded in {time.time() - start_time} seconds")

        replace_model_name = "kykim/bert-kor-base"
        replace_tokenizer = AutoTokenizer.from_pretrained(replace_model_name)
        replace_model = AutoModelForMaskedLM.from_pretrained(replace_model_name)
        print(f"Replace model loaded in {time.time() - start_time} seconds")

        model_name = "monologg/kobert"
        choice_model = BertModel.from_pretrained(model_name)
        choice_tokenizer = BertTokenizer.from_pretrained(model_name)
        print(f"Choice model loaded in {time.time() - start_time} seconds")


        ckpt_converter = './checkpoints_v2/converter'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
        tts_model = TTS(language="KR", device=device)
        print(f"TTS model loaded in {time.time() - start_time} seconds")

        separator = Separator('spleeter:2stems')
        print(f"Separator loaded in {time.time() - start_time} seconds")

        models_initialized = True
    else:
        print("Models are already initialized")

# 디렉터리 경로
video_path = "./video"
temp_path = "./temp"
audio_path = "./audio"
output_path = "./static/output"
target_dict = ['시발새끼', "씨발새끼", "씨발", "존나", "병신새끼", "새끼", '씨발 새끼', 
               '시발 새끼', '개새끼', '개 새끼','좆같은데', '족 같은데', '족같은데', '썅년', '지랄', 
               '좆까', '조까', '족가', '미친', '미친새끼', '미친 새끼', '등신', '병신', '병신 새끼']

def get_target_dict_internal():
    return target_dict

