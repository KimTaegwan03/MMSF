from whisper_transform import *
import json
import os
import pandas as pd
from tqdm import tqdm
import torch

def save_asr_result(audio_path):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    json_path = "data/asr/" + base_name + ".json"
    
    if os.path.exists(json_path):
        return 0
    
    word_list = ASR(audio_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(word_list, f, ensure_ascii=False, indent=2)

    print(f"Saved ASR result to {json_path}")
    
def save_sd_result(audio_path):
    # audio_path에서 파일명 추출 → 확장자 제거 → .json 추가
    base_name = os.path.splitext(os.path.basename(audio_path))[0]  # "00001"
    json_path = "data/sd/" + base_name + ".json"  # "00001.json"
    
    if os.path.exists(json_path):
        return 0
    
    with torch.no_grad():
        word_list = SD(audio_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(word_list, f, ensure_ascii=False, indent=2)

    print(f"Saved ASR result to {json_path}")
    
def save_json_from_words(words,base_name,result_path):
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(words,f,indent=4)
    
    print(f"TXT saved to: {result_path}")
    
    
if __name__ == "__main__":
    # for f in tqdm(kr_list,desc="ASR Processing"):
    #     save_asr_result(MSDWILD_PATH+f'/{str(f).zfill(5)}.wav')
        
    # for f in tqdm(zh_list,desc="ASR Processing"):
    #     save_asr_result(MSDWILD_PATH+f'/{str(f).zfill(5)}.wav')
    # ami_list = os.listdir("amicorpus")
    
    save_asr_result('trump.wav')
    save_sd_result('trump.wav')
    
    asr_json_path = os.path.join("data/asr/trump.json")
    sd_json_path = os.path.join("data/sd/trump.json")
    
    with open(asr_json_path, "r", encoding="utf-8") as f:
        asr_output = json.load(f)
        
    with open(sd_json_path, "r", encoding="utf-8") as f:
        sd_output = json.load(f)
        
    orch_output = orchestration(asr_output,sd_output)
    
    save_json_from_words(orch_output,'trump','data/orch/trump.json')