import whisper
from pyannote.audio import Pipeline
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import dotenv
import torch
import os


dotenv.load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API"))

HF_TOKEN = os.getenv("HF_TOKEN")
MSDWILD_PATH = "/home/dsl/data/msdwild_wavs/wav"
AMI_PATH = "./amicorpus"
ZH_FILELIST = "zh_filelist.csv"
EN_FILELIST = "en_filelist.csv"
KR_FILELIST = "kr_files.txt"

# Whisper 모델 로드 (base, small, medium, large 중 선택 가능)
asr_model = whisper.load_model("medium",device='cuda')

# diarization pipeline 불러오기
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
pipeline.to(torch.device("cuda"))

# model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2",device="cuda:1")

def ASR(audio_path):
    # 오디오 파일 디코드 및 디텍션
    result = asr_model.transcribe(audio_path, word_timestamps=True, verbose=False)
    
    # 단어 정보 추출
    word_list = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            word_info = {
                "word": word["word"].strip(),
                "start": word["start"],
                "end": word["end"]
            }
            word_list.append(word_info)
            

    return word_list


def SD(audio_path):
    # diarization 수행
    diarization = pipeline(audio_path)

    # 결과 저장
    result = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        result.append({
            "speaker": speaker,           # 예: "SPEAKER_00"
            "start": turn.start,          # float (초 단위)
            "end": turn.end               # float (초 단위)
        })

    return result

def orchestration(words, speakers):
    assigned = []

    for word in words:
        w_start, w_end = word["start"], word["end"]
        w_center = (w_start + w_end) / 2

        best_match = None
        max_overlap = 0

        for spk in speakers:
            s_start, s_end = spk["start"], spk["end"]
            # 겹치는 시간 계산
            overlap = max(0, min(w_end, s_end) - max(w_start, s_start))

            if overlap > max_overlap:
                max_overlap = overlap
                best_match = spk

        # 겹치는 화자가 없다면 가장 가까운 화자 선택
        if max_overlap == 0:
            closest_spk = min(
                speakers,
                key=lambda spk: abs(((spk["start"] + spk["end"]) / 2) - w_center)
            )
            best_match = closest_spk

        assigned.append({
            "word": word["word"],
            "start": w_start,
            "end": w_end,
            "speaker": best_match["speaker"]
        })

    return assigned

def sd_based_orchestration(whisper_words, pyannote_segments):
    assigned_segments = []

    for seg in pyannote_segments:
        seg['words'] = []

    for word in whisper_words:
        word_mid = (word['start'] + word['end']) / 2
        matched = False

        # 1단계: 겹치는 pyannote 구간 찾기 (start < end)
        for seg in pyannote_segments:
            if not (word['end'] < seg['start'] or word['start'] > seg['end']):
                seg['words'].append(word)
                matched = True
                break

        if not matched:
            # 2단계: 가장 가까운 segment에 매핑
            closest_seg = min(
                pyannote_segments,
                key=lambda seg: min(abs(word_mid - seg['start']), abs(word_mid - seg['end']))
            )
            closest_seg['words'].append(word)

    # 최종 정리
    for seg in pyannote_segments:
        seg_text = ' '.join(w['word'] for w in seg['words'])
        assigned_segments.append({
            'speaker': seg['speaker'],
            'start': seg['start'],
            'end': seg['end'],
            'words': seg['words'],
            'word': seg_text
        })

    return assigned_segments

def convert_to_llm_format(assigned_words):
    result = []
    current_speaker = None
    speaker_map = {}
    speaker_counter = 1

    for word_info in assigned_words:
        spk_raw = word_info["speaker"]

        # speaker 라벨 정규화 (<Speaker1>, <Speaker2>, ...)
        if spk_raw not in speaker_map:
            speaker_map[spk_raw] = f"<Speaker{speaker_counter}>"
            speaker_counter += 1

        spk_tag = speaker_map[spk_raw]

        # 화자가 바뀌면 새 태그로 시작
        if spk_tag != current_speaker:
            result.append(f"{spk_tag}:")
            current_speaker = spk_tag

        # 단어 추가
        result.append(word_info["word"])

    return " ".join(result)

def llm_generation(input_text, model_name="models/gemini-2.5-flash-preview-05-20",temperature=1.0): # gemini-2.5-pro-preview-06-05 , models/gemini-1.5-pro-latest, models/gemini-2.5-flash-preview-05-20
    prompt = f"""
You are a transcription editor.
You will be given a transcript that has already been processed by an automatic speech recognition (ASR) model and a speaker diarization system.
Your job is to correct any contextually incorrect, semantically awkward, or misattributed segments in the transcript, while preserving the natural characteristics of spoken language.

These mistakes may include:

Missing or incorrect words that break the meaning or flow of the sentence (e.g., misrecognized words, unnatural phrasing).

Incorrect speaker attribution (e.g., based on the context, this word or phrase should have been spoken by Speaker2, not Speaker1).

Awkward or grammatically flawed word usage that clearly detracts from the sentence meaning.

Please preserve natural spoken-language features, such as:

Contractions (e.g., "I’m", "we’re", "don’t")

Informal expressions (e.g., "yeah", "uh", "you know")

Sentence fragments and repetitions when appropriate in dialogue

Pauses or hesitations (you can represent them with commas or ellipses, if they help readability)

Do not rewrite fluent or accurate portions. Only fix what is clearly incorrect, misleading, or awkward.

Output format:
Use the following format strictly:
[Speaker1] corrected sentence [Speaker2] corrected sentence [Speaker1] corrected sentence ...

Transcript:
{input_text}
"""
    # prompt = f"""
    # In the transcript below, there are unnatural speaker assignments or speech recognition errors based on the context.
    # Please revise the speaker labels and the wording to make the dialogue more natural and contextually appropriate.
    # Return only the corrected result and do not explain the reasons for the changes.
    
    # Use the following format strictly:
    # [Speaker1] corrected sentence [Speaker2] corrected sentence [Speaker3] corrected sentence ...
    
    # Transcript:
    # {input_text}
    # """
    generation_config = genai.GenerationConfig(temperature=temperature)
    model = genai.GenerativeModel(model_name,generation_config=generation_config)
    response = model.generate_content(prompt)

    return response.text.strip()

import re

def parsing_llm_output(text):
    # SPEAKER + 문장 매핑용 정규식
    pattern = r'\[(Speaker\d+)\]\s*([^[]+)'
    matches = re.findall(pattern, text)

    result = []
    for speaker, utterance in matches:
        utterance = utterance.strip()
        if utterance:
            result.append({
                "word": utterance,
                "speaker": speaker
            })

    return result

def llm_output_to_diarizationLM_format(orche_output, llm_output):
    src_text = []   # llm text
    src_spk = []    # llm speaker
    tgt_text = []   # orche text
    tgt_spk = []    # orche speaker
    
    pattern = r'SPEAKER_(\d+)'# SPEAKER_01
    pattern2 = r'Speaker(\d+)'
    
    for i in range(len(orche_output)): # word speaker start end
        match = re.match(pattern, orche_output[i]['speaker'])
        number = int(match.group(1))
        tgt_text.append(orche_output[i]['word'].replace(' ',''))
        tgt_spk.append(str(number))
        
    for i in range(len(llm_output)):
        match = re.match(pattern2, llm_output[i]['speaker'])
        number = int(match.group(1))
        src_text.append(llm_output[i]['word'].replace(' ',''))
        src_spk.append(str(number))
        
    src_text = ' '.join(src_text)
    src_spk = ' '.join(src_spk)
    tgt_text = ' '.join(tgt_text)
    tgt_spk = ' '.join(tgt_spk)
        
    return src_text, src_spk, tgt_text, tgt_spk

def transferred_spk_to_output(orche_output, spk:str):
    output = []
    spk = spk.split(' ')
    
    for i in range(len(orche_output)):
        output.append({
            "word": orche_output[i]['word'],
            "start": orche_output[i]['start'],
            "end": orche_output[i]['end'],
            "speaker": spk[i]
        })
    
    return output
    

def levenshtein_align(sent1, sent2):
    words1 = sent1
    words2 = sent2
    len1 = len(words1)
    len2 = len(words2)

    # DP 초기화
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    back = [[None] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
        back[i][0] = 'up'
    for j in range(len2 + 1):
        dp[0][j] = j
        back[0][j] = 'left'

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = 'diag'
            else:
                options = [
                    (dp[i - 1][j] + 1, 'up'),      # 삭제
                    (dp[i][j - 1] + 1, 'left'),    # 삽입
                    (dp[i - 1][j - 1] + 1, 'diag') # 치환
                ]
                dp[i][j], back[i][j] = min(options)

    # 역추적
    i, j = len1, len2
    alignments = []
    while i > 0 or j > 0:
        direction = back[i][j]
        if direction == 'diag':
            alignments.append((i - 1, j - 1))  # (index_in_input1, index_in_input2)
            i -= 1
            j -= 1
        elif direction == 'up':
            alignments.append((i - 1, None))  # 입력1의 단어가 삭제됨
            i -= 1
        elif direction == 'left':
            alignments.append((None, j - 1))  # 입력2에만 존재하는 삽입 단어
            j -= 1

    alignments.reverse()

    # 입력1 기준 매핑 리스트 생성
    mapping1 = [None] * len1
    # 입력2 기준 매핑 리스트
    mapping2 = [None] * len2
    for i1, i2 in alignments:
        if i1 is not None:
            mapping1[i1] = i2
        if i2 is not None:
            mapping2[i2] = i1

    return mapping1, mapping2

def cosine_distance(v1, v2):
    return 1 - cosine_similarity([v1], [v2])[0, 0]  # 코사인 거리 = 1 - 유사도

# def dtw_align_with_embeddings(sent1, sent2, window=1000):
#     len1, len2 = len(sent1), len(sent2)
#     emb1 = model.encode(sent1)
#     emb2 = model.encode(sent2)

#     if window is None:
#         window = max(len1, len2)  # 제한 없음

#     cost = np.full((len1 + 1, len2 + 1), np.inf)
#     backtrack = np.empty((len1 + 1, len2 + 1), dtype=object)
#     cost[0, 0] = 0

#     def cosine_distance(v1, v2):
#         return 1 - cosine_similarity([v1], [v2])[0, 0]

#     for i in range(1, len1 + 1):
#         j_start = max(1, i - window)
#         j_end = min(len2 + 1, i + window + 1)
#         for j in range(j_start, j_end):
#             d = cosine_distance(emb1[i - 1], emb2[j - 1])
#             options = []
#             if cost[i - 1, j - 1] != np.inf:
#                 options.append((cost[i - 1, j - 1], 'diag'))
#             if cost[i - 1, j] != np.inf:
#                 options.append((cost[i - 1, j], 'up'))
#             if cost[i, j - 1] != np.inf:
#                 options.append((cost[i, j - 1], 'left'))
#             if options:
#                 min_cost, direction = min(options)
#                 cost[i, j] = d + min_cost
#                 backtrack[i, j] = direction

#     # Backtrack
#     i, j = len1, len2
#     alignments = []
#     while i > 0 and j > 0:
#         direction = backtrack[i, j]
#         if direction == 'diag':
#             alignments.append((i - 1, j - 1))
#             i -= 1
#             j -= 1
#         elif direction == 'up':
#             alignments.append((i - 1, None))
#             i -= 1
#         elif direction == 'left':
#             alignments.append((None, j - 1))
#             j -= 1
#         else:
#             break

#     alignments.reverse()

#     mapping1 = [None] * len1
#     mapping2 = [None] * len2
#     for i1, i2 in alignments:
#         if i1 is not None:
#             mapping1[i1] = i2
#         if i2 is not None:
#             mapping2[i2] = i1

#     return mapping1, mapping2


def split_to_word_speaker_w_time(chunks):
    word_list = []
    speaker_list = []
    time_list = []
    speaker_map = {}
    speaker_id = 1

    for chunk in chunks:
        sentence = chunk['word']
        speaker = chunk['speaker']
        start = chunk['start']
        end = chunk['end']
        if speaker not in speaker_map:
            speaker_map[speaker] = speaker_id
            speaker_id += 1
        speaker_num = speaker_map[speaker]

        # 단어 기준 분리 (공백 기준)
        words = sentence.strip().split()
        word_list.extend(words)
        speaker_list.extend([speaker_num] * len(words))
        time_list.append([start,end])

    return word_list, speaker_list, time_list

def split_to_word_speaker(chunks):
    word_list = []
    speaker_list = []
    speaker_map = {}
    speaker_id = 1

    for chunk in chunks:
        sentence = chunk['word']
        speaker = chunk['speaker']
        if speaker not in speaker_map:
            speaker_map[speaker] = speaker_id
            speaker_id += 1
        speaker_num = speaker_map[speaker]

        # 단어 기준 분리 (공백 기준)
        words = sentence.strip().split()
        word_list.extend(words)
        speaker_list.extend([speaker_num] * len(words))

    return word_list, speaker_list

def merge_close_segments(output, k=0.2):
    """
    output: [{'word': str, 'start': float, 'end': float, 'speaker': str}, ...]
    k: float, 같은 화자의 단어 사이가 k초 이하이면 붙임

    return: 수정된 output 리스트
    """
    if not output:
        return []

    new_output = [output[0]]

    for i in range(1, len(output)):
        prev = new_output[-1]
        curr = output[i]

        if prev['speaker'] == curr['speaker']:
            gap = curr['start'] - prev['end']
            if gap <= k:
                # 현재 단어의 시작 시점을 이전 단어의 종료 시점으로 붙임
                curr = curr.copy()
                curr['start'] = prev['end']

        new_output.append(curr)

    return new_output

def write_rttm(file_list, all_segment_lists, output_path="output.rttm"):
    with open(output_path, "w", encoding="utf-8") as f:
        for file_id, segments in zip(file_list, all_segment_lists):
            file_id = file_id.replace(".wav", "")  # RTTM에서 확장자 제거

            for seg in segments:
                start = float(seg["start"])
                end = float(seg["end"])
                duration = end - start
                speaker = seg["speaker"]

                rttm_line = f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
                f.write(rttm_line)

    print(f"RTTM saved to: {output_path}")

def load_file_list(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    import json
    from diarizationLM_test import save_rttm_from_words
    
    zh_list = pd.read_csv(ZH_FILELIST)["filename"].to_list()
    en_list = pd.read_csv(EN_FILELIST)["filename"].to_list()
    kr_list = load_file_list(KR_FILELIST)
    
    for f in tqdm(en_list):
        base_name = str(f).zfill(5)
        asr_json_path = "data/asr_x/" + base_name + ".json"
        sd_json_path = "data/sd/" + base_name + ".json"
        
        with open(asr_json_path, "r", encoding="utf-8") as f:
            asr_output = json.load(f)
            
        with open(sd_json_path, "r", encoding="utf-8") as f:
            sd_output = json.load(f)
            
        orch_output = orchestration(asr_output,sd_output)
        
        save_rttm_from_words(orch_output, base_name, f"result/orchestration/en_rttm/{base_name}.rttm")