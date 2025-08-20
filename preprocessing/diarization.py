from whisper_transform import *
import json
import os
import pandas as pd
from tqdm import tqdm
import torch
import subprocess
import sqlite3

import logging
import traceback

logging.basicConfig(
    filename='error.log',        
    level=logging.ERROR,          
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'                 
)

def save_asr_result(audio_path,output_path):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    json_path = "data/asr/" + base_name + ".json"
    
    if os.path.exists(json_path):
        return 0
    
    word_list = ASR(audio_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(word_list, f, ensure_ascii=False, indent=2)

    print(f"Saved ASR result to {output_path}")
    
def save_sd_result(audio_path,output_path):
    
    base_name = os.path.splitext(os.path.basename(audio_path))[0]  # "00001"
    json_path = "data/sd/" + base_name + ".json"  # "00001.json"
    
    if os.path.exists(json_path):
        return 0
    
    with torch.no_grad():
        word_list = SD(audio_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(word_list, f, ensure_ascii=False, indent=2)

    print(f"Saved ASR result to {output_path}")
    
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

    os.makedirs("data/asr",exist_ok=True)
    os.makedirs("data/sd",exist_ok=True)
    os.makedirs("data/orch",exist_ok=True)

    db_path = "/mnt/share65/speech_segments.db"

    video_dir = "/mnt/share65/videos"
    json_dir = "/mnt/share65/orch_jsons"
    asr_dir = "/mnt/share65/asr"
    sd_dir = "/mnt/share65/sd"

    os.makedirs(json_dir,exist_ok=True)
    os.makedirs(asr_dir,exist_ok=True)
    os.makedirs(sd_dir,exist_ok=True)

    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query("SELECT video_id FROM video_metadata ORDER BY video_id", conn)

    video_list = df['video_id'].to_list()
    video_list.sort()

    for file in tqdm(video_list,desc="Processing Video..."):

        try:
            # if not file.endswith(".mp4"):
            #     continue

            input_video = video_dir + "/" + file + ".mp4"
            basename = file #.replace(".mp4","")

            if os.path.exists(os.path.join(json_dir,f"{basename}.json")):
                continue
            
            command = [
                "ffmpeg",
                "-i", input_video,
                "-y",                
                "-vn",                
                "-acodec", "pcm_s16le",  
                "-ar", "16000",       
                "-ac", "1",           
                "temp_audio.wav"
            ]

            subprocess.run(command, check=True)

            asr_json_path = os.path.join(asr_dir,basename+".json")
            sd_json_path = os.path.join(sd_dir,basename+".json")

            if not os.path.exists(asr_json_path):
                save_asr_result('temp_audio.wav',asr_json_path)      # json_path = "data/sd/" + base_name + ".json"  # "00001.json"

            if not os.path.exists(sd_json_path):
                save_sd_result('temp_audio.wav',sd_json_path)
        
            
            with open(asr_json_path, "r", encoding="utf-8") as f:
                asr_output = json.load(f)
                
            with open(sd_json_path, "r", encoding="utf-8") as f:
                sd_output = json.load(f)
                
            orch_output = orchestration(asr_output,sd_output)
            
            save_json_from_words(orch_output,basename,os.path.join(json_dir,f"{basename}.json"))
            
        except Exception as e:
            error_details = traceback.format_exc()
            logging.error("An error occurred:\n%s", error_details)
