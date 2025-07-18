import torch
import pandas as pd
from transformers import BeitImageProcessor, BeitModel, AutoProcessor, HubertModel, pipeline, AutoFeatureExtractor, AutoTokenizer, AutoModel
from PIL import Image
import re
import librosa
from moviepy.editor import VideoFileClip
import io
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List
import subprocess
import soundfile as sf
import cv2
import json
from feature_extractor import TextFeatureExtractor, VideoFeatureExtractor, AudioFeatureExtractor
from torch_geometric.data import Data
import math

from pyvis.network import Network
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_tree_edge_index(num_nodes, branch_factor,level_edge=True):
    if branch_factor < 1:
        raise ValueError("branch_factor는 1 이상이어야 합니다.")
    if num_nodes < 2:
        raise ValueError("노드 수는 최소 2개 이상이어야 트리 구조가 가능합니다.")

    # 1. 트리 depth 계산
    depth = 1
    total = 1
    prev = 1
    while total < num_nodes:
        total += (prev-1) * (branch_factor-1) + branch_factor
        depth += 1

    print("Depth:", depth, "Total Nodes:", total)

    # 실제 생성 가능한 노드 수를 초과한 경우, 마지막 층 일부는 잘릴 수 있음
    level_nodes = [[0]]
    node_id = 1
    previous_count = 1
    for d in range(1,depth):
        count = (previous_count-1) * (branch_factor-1) + branch_factor
        previous_count = count
        layer = []
        for _ in range(count):
            if node_id >= num_nodes:
                break
            layer.append(node_id)
            node_id += 1
        level_nodes.append(layer)
        if node_id >= num_nodes:
            break

    # 2. edge 구성 (상위 노드 → 하위 노드)
    edges = []
    for l in range(len(level_nodes) - 1):
        parents = level_nodes[l]
        children = level_nodes[l + 1]
        for i, p in enumerate(parents):
            if i != 0 and level_edge:  # 첫 번째 부모 노드는 제외
                edges.append((p, parents[i - 1]))
                edges.append((parents[i - 1], p))
            for j in range(branch_factor):
                c_index = i + j
                if c_index < len(children):
                    edges.append((p, children[c_index]))
                    edges.append((children[c_index], p))

    # 마지막 층의 노드가 부족한 경우, 마지막 층의 노드끼리 연결
    if level_edge and len(level_nodes[-1]) > 1:
        last_layer = level_nodes[-1]
        for i in range(len(last_layer) - 1):
            edges.append((last_layer[i], last_layer[i + 1]))
            edges.append((last_layer[i + 1], last_layer[i]))

    # 3. edge_index 반환
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return edge_index

def convert_webm_to_mp4(video_path, converted_video_path):
    """OpenCV를 사용하여 WebM을 MP4로 변환"""
    # print("WebM을 MP4로 변환 중...")
    clip = VideoFileClip(video_path)
    clip.write_videofile(converted_video_path, codec="libx264", audio_codec="aac")

def download_video(url, output_path, i):
    """
    주어진 url에서 비디오를 다운로드하여 출력 경로에 저장합니다.

    매개변수:
    url (str): 다운로드할 비디오의 URL입니다.
    output_path (str): 비디오를 저장할 경로입니다.

    반환값:
    dict: 비디오의 메타데이터가 포함된 사전입니다.
    """
    # 경로 생성
    os.makedirs(output_path, exist_ok=True)

    # 비디오 재생 시간 확인
    command = f'yt-dlp --get-duration {url}'
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return 3601
    duration = result.stdout.strip()

    # 재생 시간이 1시간 이상인 경우 함수 종료
    time_parts = duration.split(':')
    if len(time_parts) == 3:
        hours, minutes, seconds = map(int, time_parts)
    elif len(time_parts) == 2:
        hours = 0
        minutes, seconds = map(int, time_parts)
    elif len(time_parts) == 1:
        hours = 0
        minutes = 0
        seconds = int(time_parts[0])
    else:
        raise ValueError("Unexpected duration format")
    total_seconds = hours * 3600 + minutes * 60 + seconds

    if total_seconds > 3600:
        return total_seconds
    
    else:
        # yt-dlp 명령 실행
        try:
            command = f'yt-dlp --force-overwrites -f "bestvideo[height=240]+bestaudio" --cookies-from-browser firefox --user-agent "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0" -o "{output_path}/video_{i}" {url}'
            subprocess.run(command, shell=True, check=True)

            return total_seconds
        except:
            return 3601
        
def merge_words_by_speaker(word_list):
    """
    같은 화자의 연속된 단어들을 문장 단위로 병합하고, speaker를 숫자 ID로 변환.

    Parameters:
        word_list (list): 각 단어에 대한 정보가 담긴 딕셔너리 리스트

    Returns:
        list: 문장 단위로 병합된 딕셔너리 리스트 (speaker는 int ID)
    """
    if not word_list:
        return []

    def speaker_to_id(speaker_str):
        match = re.search(r'\d+', speaker_str)
        return int(match.group()) if match else -1

    merged = []
    current = {
        "speaker": speaker_to_id(word_list[0]["speaker"]),
        "word": word_list[0]["word"],
        "start": word_list[0]["start"],
        "end": word_list[0]["end"]
    }

    for word_info in word_list[1:]:
        speaker_id = speaker_to_id(word_info["speaker"])
        if speaker_id == current["speaker"]:
            current["word"] += " " + word_info["word"]
            current["end"] = word_info["end"]
        else:
            merged.append(current)
            current = {
                "speaker": speaker_id,
                "word": word_info["word"],
                "start": word_info["start"],
                "end": word_info["end"]
            }

    merged.append(current)
    return merged

def get_model_memory_usage(model):
    """ 모델이 차지하는 GPU 메모리 크기 출력 (GB 단위) """
    model_parameters = sum(p.numel() for p in model.parameters())  # 총 파라미터 개수
    model_memory_bytes = model_parameters * 4  # float32(4 bytes) 기준
    model_memory_gb = model_memory_bytes / (1024 ** 3)  # GB 변환

    print(f"Model Parameters: {model_parameters:,}")  # 1000 단위 콤마 추가
    print(f"Model GPU Memory Usage: {model_memory_gb:.6f} GB")

def print_gpu_memory_usage():
    """ 현재 할당된 모델의 GPU 메모리 사용량 출력 """
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB 단위 변환
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB 단위 변환

    print(f" 현재 할당된 GPU 메모리: {allocated:.4f} GB") 
    print(f" 현재 예약된 GPU 메모리: {reserved:.4f} GB")

class TextFeatureExtractor:
    def __init__(self):
        self.name = "BERTTextFeatureExtractor"

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.model.to(device)

    def encode(self, text:str):
        # 토큰화 및 텐서 변환
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

        # 모델을 통해 임베딩 추출
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 마지막 히든 스테이트 추출
        last_hidden_states = outputs.last_hidden_state  # shape: (batch_size, sequence_length, hidden_size)


        # 문장 임베딩으로 [CLS] 토큰의 벡터 사용
        sentence_embedding = last_hidden_states[:, 0, :]  # shape: [batch_size, 768]
        # sentence_embedding = last_hidden_states.mean(dim=1)  # shape: [batch_size, 768]

        return sentence_embedding
        

class VideoFeatureExtractor:
    def __init__(self, model="microsoft/beit-base-patch16-224"):
        self.name = "BEiT3VideoFeatureExtractor"
        
        # BEiT-3 모델과 이미지 프로세서 로드
        self.processor = BeitImageProcessor.from_pretrained(model)
        self.model = BeitModel.from_pretrained(model)
        self.model.to(device)

    def encode(self, image: Image.Image):
        """
        이미지에서 특징을 추출하는 함수입니다.
        
        Args:
            image (PIL.Image): 입력 이미지
            
        Returns:
            torch.Tensor: 추출된 특징 벡터
        """
        # 이미지 전처리
        inputs = self.processor(images=image, return_tensors="pt").to(device)

        # 모델을 통해 특징 추출
        with torch.no_grad():
            outputs = self.model(**inputs) # [batch_size, sequence_length, hidden_size]
        
        # 최종 특징 벡터 (CLS 토큰 사용)
        features = outputs.last_hidden_state[:, 0, :]
        return features
    
    def aggregate_encoding_feature(self, frames):
        """
        주어진 프레임 리스트를 인코딩하고 평균 풀링을 통해 특징을 집계합니다.

        Args:
            frames (list[PIL.Image]): PIL Image 객체로 구성된 리스트

        Returns:
            torch.Tensor: 모든 프레임의 특징을 평균 풀링한 결과
        """
        encoded_features = [self.encode(frame) for frame in frames]
        return torch.mean(torch.stack(encoded_features), dim=0)

    def extract_from_video(self, video_frames_folder):
        """
        비디오 프레임 폴더에서 프레임을 읽고 특징을 추출합니다.

        Args:
            video_frames_folder (str): 프레임 이미지가 저장된 폴더 경로

        Returns:
            torch.Tensor: 집계된 특징 벡터
        """
        frame_files = sorted([f for f in os.listdir(video_frames_folder) if f.endswith(".jpg") or f.endswith(".png")])
        frames = [Image.open(os.path.join(video_frames_folder, file)).convert("RGB") for file in frame_files]
        
        return self.aggregate_encoding_feature(frames)

    def save_features(self, features, output_path):
        """
        추출된 특징 벡터를 저장합니다.

        Args:
            features (torch.Tensor): 저장할 특징 벡터
            output_path (str): 저장할 파일 경로
        """
        torch.save(features, output_path)
        print(f"Features saved to {output_path}")
        
    def save_frames_every_n_seconds(self, video_path, output_dir, interval_sec=1):
        """
        비디오에서 n초마다 프레임을 저장하는 함수.

        Parameters:
            video_path (str): 비디오 파일 경로
            output_dir (str): 프레임을 저장할 디렉토리
            interval_sec (int or float): 프레임 추출 간격 (초 단위)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"Video duration: {duration:.2f} seconds, FPS: {fps:.2f}")

        frame_interval = int(fps * interval_sec)
        frame_idx = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                time_sec = frame_idx / fps
                filename = f"frame_{time_sec:.2f}s.png"
                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, frame)
                saved_count += 1

            frame_idx += 1

        cap.release()
        print(f"Saved {saved_count} frames every {interval_sec} seconds to '{output_dir}'")

class AudioFeatureExtractor:
    def __init__(self, model="facebook/hubert-base-ls960",audio_folder="output/split_audio"):
        self.name = "AudioFeatureExtractor"
        self.audio_folder = audio_folder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # HuBERT 모델과 프로세서 로드
        self.processor = AutoFeatureExtractor.from_pretrained(model)
        self.model = HubertModel.from_pretrained(model).to(self.device)
        self.model.eval()
        self.model.half()
        # self.model = pipeline("feature-extraction", model="facebook/hubert-base-ls960",torch_dtype=torch.float16,device="cuda")

    def encode(self, data, sampling_rate=16000):
        """
        오디오 데이터를 특징 벡터로 변환하는 함수입니다.
        
        Args:
            data (Tensor): 오디오 데이터를 나타내는 Tensor
            sampling_rate (int): 샘플링 레이트 (기본값: 16000Hz)
        
        Returns:
            torch.Tensor: 추출된 특징 벡터
        """
        # 오디오 데이터를 HuBERT 프로세서에 맞게 전처리
        inputs = self.processor(data, return_tensors="pt", sampling_rate=sampling_rate).to(self.device)
         
        # 모델을 통해 특징 추출
        with torch.no_grad():
            outputs = self.model(**inputs.to(torch.float16))
        
        # 마지막 hidden state에서 특징 벡터 추출 [CLS]
        features = outputs.last_hidden_state
        return features[:,0,:]

    def extract_from_audio_folder(self, audio_path):
        """
        분할된 오디오 파일을 불러와 특징을 추출하고 저장합니다.
        """
        audio_files = sorted([f for f in os.listdir(audio_path) if f.endswith(".wav")])

        features = []
        
        for audio_file in audio_files:
            file_path = os.path.join(audio_path, audio_file)
            # print(f"Processing: {file_path}")
            
            # 오디오 파일 로드
            audio_data, sr = librosa.load(file_path, sr=16000)
            audio_tensor = torch.tensor(audio_data).to(self.device)  # (1, Samples)
            
            # 특징 벡터 추출
            features.append(self.encode(audio_tensor, sr))

        return torch.mean(torch.stack(features), dim=0)
    
# class MultimodalFeatureExtractor:
#     def __init__(self):
#         self.model = ImageBindModel.from_pretrained("nielsr/imagebind-huge")

#     def extract_features(self, text, video_frames_folder, audio_folder):
#         text_features = self.text_extractor.encode(text)
#         video_features = self.video_extractor.extract_from_video(video_frames_folder)
#         audio_features = self.audio_extractor.extract_from_audio_folder(audio_folder)

#         return text_features, video_features, audio_features

def json_to_multi_modal_embedding(json_dir, video_dir, audio_dir, output_dir):
    """
    JSON 파일을 읽어 멀티모달 임베딩을 생성하고 저장합니다.
    
    Args:
        json_dir (str): JSON 파일이 저장된 디렉토리
        video_dir (str): 비디오 프레임이 저장된 디렉토리
        output_dir (str): 결과를 저장할 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)

    text_extractor = TextFeatureExtractor()
    video_extractor = VideoFeatureExtractor()
    audio_extractor = AudioFeatureExtractor()
    
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])

    for json_file in tqdm(json_files, desc="Processing JSON files"):
        basename = os.path.basename(json_file).replace(".json", "")
        json_path = os.path.join(json_dir, json_file)
        video_path = os.path.join(video_dir, f"{basename}.mp4")

        if not os.path.exists(video_path):
            convert_webm_to_mp4(os.path.join(video_dir, f"{basename}.webm"), video_path)

        input_video = video_path
        output_audio = audio_dir + f"/{basename}.wav"

        if not os.path.exists(output_audio):
            command = [
                "ffmpeg",
                "-i", input_video,
                "-y",                # 기존 파일 덮어쓰기
                "-vn",                # 비디오 제외
                "-acodec", "pcm_s16le",  # WAV 포맷 (16-bit)
                "-ar", "16000",       # 샘플링 레이트 16kHz
                "-ac", "1",           # 모노 오디오
                output_audio
            ]

            subprocess.run(command, check=True)

        # JSON 파일 읽기
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            
        json_data = merge_words_by_speaker(json_data)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("비디오 파일을 열 수 없습니다.")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)

        for idx, item in enumerate(json_data):
            print(f"Processing item {idx+1}/{len(json_data)}")
            word = item['word']
            speaker = item['speaker']
            start = item['start']
            end = item['end']

            if (end - start) < 1:
                continue

            # 비디오 프레임 추출
            start_frame = int(start * fps)
            end_frame = int(end * fps)

            frames = []
            for frame_num in range(start_frame, end_frame, int(fps)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            # 프레임을 PIL 이미지로 변환
            pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]

            # 특징 추출
            print("Processing video features...")
            video_features = video_extractor.aggregate_encoding_feature(pil_frames)

            # 결과 저장
            video_output_dir = os.path.join(f"{output_dir}/video_features",basename)
            output_path = os.path.join(video_output_dir, f"{speaker}_{start:.2f}_{end:.2f}.pt")

            os.makedirs(video_output_dir, exist_ok=True)
            torch.save(video_features, output_path)

            ### 오디오 분할 및 저장, 임베딩 ###
            y, sr = librosa.load(output_audio, sr=16000)

            start_sample = int(start * 16000)
            end_sample = int(end * 16000)

            utterance_audio = y[start_sample:end_sample]

            print("Processing audio features...")
            audio_features = audio_extractor.encode(utterance_audio, sr)

            audio_output_dir = os.path.join(f"{output_dir}/audio_features",basename)
            output_path = os.path.join(audio_output_dir, f"{speaker}_{start:.2f}_{end:.2f}.pt")

            os.makedirs(audio_output_dir, exist_ok=True)
            torch.save(audio_features, output_path)
            
            ######

            ### 텍스트 임베딩 ###
            print("Processing text features...")
            text_features = text_extractor.encode(word)
            text_output_dir = os.path.join(f"{output_dir}/text_features",basename)
            output_path = os.path.join(text_output_dir, f"{speaker}_{start:.2f}_{end:.2f}.pt")

            os.makedirs(text_output_dir, exist_ok=True)
            torch.save(text_features, output_path)
            ######

        cap.release()

def construct_conversation_graph(json_dir, data_dir, output_dir, draw=False):
    """
    JSON 파일을 읽어 대화 그래프를 구성하고 저장합니다.
    
    Args:
        json_dir (str): JSON 파일이 저장된 디렉토리
        output_path (str): 결과를 저장할 파일 경로
    """

    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])

    for json_file in tqdm(json_files, desc="Constructing conversation graph"):
        conversation_graph = Data()
        
        with open(os.path.join(json_dir, json_file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        data = merge_words_by_speaker(data)

        text_features = []
        video_features = []
        audio_features = []

        speaker_num = []

        offset = 0  # 노드 인덱스 오프셋 초기화

        for item in tqdm(data, desc="Processing utterances"):
            word = item['word']
            speaker = item['speaker']
            start = item['start']
            end = item['end']

            if not word or (end - start) < 1:
                offset += 1
                continue

            speaker_num.append(speaker)

            # 텍스트 임베딩
            with open(os.path.join(data_dir, "text_features", json_file.replace(".json", f"/{speaker}_{start:.2f}_{end:.2f}.pt")), 'rb') as f:
                text_feature:torch.Tensor = torch.load(f)
                text_features.append(text_feature)
                print("Shape of text feature:", text_feature.shape)

            # 비디오 임베딩
            with open(os.path.join(data_dir, "video_features", json_file.replace(".json", f"/{speaker}_{start:.2f}_{end:.2f}.pt")), 'rb') as f:
                video_feature = torch.load(f)
                video_features.append(video_feature)
                print("Shape of video feature:", text_feature.shape)

            # 오디오 임베딩
            with open(os.path.join(data_dir, "audio_features", json_file.replace(".json", f"/{speaker}_{start:.2f}_{end:.2f}.pt")), 'rb') as f:
                audio_feature = torch.load(f)
                audio_features.append(audio_feature)
                print("Shape of audio feature:", text_feature.shape)

        # 대화 그래프에 노드 추가
        conversation_graph.x_text = torch.stack(text_features,dim=0)
        conversation_graph.x_video = torch.stack(video_features,dim=0)
        conversation_graph.x_audio = torch.stack(audio_features,dim=0)
        conversation_graph.speaker_num = torch.tensor(speaker_num, dtype=torch.long) if speaker_num else None
        conversation_graph.global_index = []
        conversation_graph.speaker_index = []

        node_count = len(text_features)  # 노드 개수
        
        speaker_color = [
    'blue',      # 1
    'green',     # 2
    'orange',    # 3
    'skyblue',   # 4
    'cyan',      # 5
    'gold',      # 6
    'lime',      # 7
    'brown',     # 8
    'pink',      # 9
    'gray'       # 10
]

        node_color = []

        # 대화 그래프에 Sequential 엣지 추가
        for idx in range(node_count):
            node_color.append('red')
            if idx == 0:
                continue
            # 이전 노드와 현재 노드 간의 엣지 추가
            elif idx == 1:
                conversation_graph.edge_index = torch.tensor([[idx - 1, idx], [idx, idx - 1]], dtype=torch.long)
            else:
                conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[idx - 1, idx], [idx, idx - 1]], dtype=torch.long)], dim=1)


        # 대화 그래프에 Hierarchical-conversation 노드 임베딩 초기화
        hie_conv_count = (node_count - 1) * node_count // 2
        conversation_graph.x_hie_conv = torch.zeros((hie_conv_count, 768), dtype=torch.float)  # Hierarchical-conversation 노드 임베딩 초기화 (추후 nn.Parameter로 교체 가능)

        conversation_graph.global_index.append(node_count)

        # 대화 그래프에 Hierarchical-conversation 엣지 추가
        edge_index_hie_conv = build_tree_edge_index(hie_conv_count, branch_factor=2)

        # conversation_graph의 edge_index와 edge_index_hie_conv를 연결
        edge_index_hie_conv = edge_index_hie_conv + node_count # offset  # Hierarchical-conversation 엣지 인덱스에 오프셋 추가
        conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, edge_index_hie_conv], dim=1)

        for i in range(hie_conv_count):
            node_color.append('purple')
    
        for i in range(node_count):
            if i == 0:
                conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[node_count - i - 1], [node_count + hie_conv_count - 1 - i]], dtype=torch.long)], dim=1)
                conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[node_count + hie_conv_count - 1 - i], [node_count - i - 1]], dtype=torch.long)], dim=1)
            elif i == node_count-1:
                conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[node_count - i - 1], [node_count + hie_conv_count - i]], dtype=torch.long)], dim=1)
                conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[node_count + hie_conv_count - i], [node_count - i - 1]], dtype=torch.long)], dim=1)
            else:
                conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[node_count - i - 1], [node_count + hie_conv_count - i]], dtype=torch.long)], dim=1)
                conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[node_count + hie_conv_count - i], [node_count - i - 1]], dtype=torch.long)], dim=1)

                conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[node_count - i - 1], [node_count + hie_conv_count - 1 - i]], dtype=torch.long)], dim=1)
                conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[node_count + hie_conv_count - 1 - i], [node_count - i - 1]], dtype=torch.long)], dim=1)

        # 대화 그래프에 Hierarchical-speaker 노드 임베딩 초기화
        # 각 화자마다 노드 개수를 구하고 Hierarchical-conversation과 유사한 방식으로 노드 임베딩을 초기화
        prev_speaker_num = 0
        unique_speakers = set(speaker_num)
        conversation_graph.x_hie_speaker = 0  # Hierarchical-speaker 노드 임베딩 초기화

        for speaker in unique_speakers:
            speaker_indices = [i for i, s in enumerate(speaker_num) if s == speaker]
            speaker_node_count = len(speaker_indices)
            if speaker_node_count < 2:
                continue
            
            # Hierarchical-speaker 노드 임베딩 초기화
            hie_speaker_count = (speaker_node_count - 1) * speaker_node_count // 2

            conversation_graph.speaker_index.append(node_count + hie_conv_count + prev_speaker_num)

            if conversation_graph.x_hie_speaker is 0:
                conversation_graph.x_hie_speaker = torch.zeros((hie_speaker_count, 768), dtype=torch.float)
            else:
                conversation_graph.x_hie_speaker = torch.concat([conversation_graph.x_hie_speaker,torch.zeros((hie_speaker_count, 768), dtype=torch.float)], dim=0)  # Hierarchical-speaker 노드 임베딩 초기화 (추후 nn.Parameter로 교체 가능)

            for i in range(hie_speaker_count):
                # print(speaker)
                node_color.append(speaker_color[speaker])
        
            if hie_speaker_count != 1:

                # Hierarchical-speaker 엣지 추가
                edge_index_hie_speaker = build_tree_edge_index(hie_speaker_count, branch_factor=2)

                # conversation_graph의 edge_index와 edge_index_hie_speaker를 연결
                edge_index_hie_speaker = edge_index_hie_speaker + node_count + hie_conv_count + prev_speaker_num # offset  # Hierarchical-speaker 엣지 인덱스에 오프셋 추가
                conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, edge_index_hie_speaker], dim=1)
                

            for i in range(len(speaker_indices)): # s는 현재 화자의 대화 노드 인덱스
                s = speaker_indices[-1-i]
                if i == 0:
                    conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[s], [node_count + hie_conv_count + hie_speaker_count + prev_speaker_num - 1 - i]], dtype=torch.long)], dim=1)
                    conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[node_count + hie_conv_count + hie_speaker_count + prev_speaker_num - 1 - i], [s]], dtype=torch.long)], dim=1)
                if i == len(speaker_indices)-1:
                    conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[s], [node_count + hie_conv_count + hie_speaker_count + prev_speaker_num  - i]], dtype=torch.long)], dim=1)
                    conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[node_count + hie_conv_count + hie_speaker_count + prev_speaker_num - i], [s]], dtype=torch.long)], dim=1)
                if i != 0 and i != len(speaker_indices)-1:
                    conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[s], [node_count + hie_conv_count + hie_speaker_count + prev_speaker_num - i]], dtype=torch.long)], dim=1)
                    conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[node_count + hie_conv_count + hie_speaker_count + prev_speaker_num - i], [s]], dtype=torch.long)], dim=1)
                    conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[s], [node_count + hie_conv_count + hie_speaker_count + prev_speaker_num - 1 - i]], dtype=torch.long)], dim=1)
                    conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[node_count + hie_conv_count + hie_speaker_count + prev_speaker_num - 1 - i], [s]], dtype=torch.long)], dim=1)

            prev_speaker_num += hie_speaker_count
        
        if draw:
            G = nx.DiGraph()

            for edge in conversation_graph.edge_index.t().tolist():
                G.add_edge(edge[0], edge[1])

            # pos = nx.spring_layout(G)  # 노드 위치 설정
            net = Network(notebook=False, height='900px', width='100%')
            net.from_nx(G)

            for node in net.nodes:
                node["label"] = str(node["id"])  # id를 문자열로 label로 사용
                node["color"] = node_color[node["id"]]  # 기존 색상 그대로
                
            net.show('graph.html',notebook=False)

        base = json_file.replace(".json", "")

        output_path = os.path.join(output_dir,f"{base}.pt")
        torch.save(conversation_graph,output_path)

def construct_conversation_graph_with_single_global(json_dir, data_dir, output_dir,draw=False):
    """
    JSON 파일을 읽어 대화 그래프를 구성하고 저장합니다.
    
    Args:
        json_dir (str): JSON 파일이 저장된 디렉토리
        output_path (str): 결과를 저장할 파일 경로
    """

    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])

    for json_file in tqdm(json_files, desc="Constructing conversation graph"):
        conversation_graph = Data()
        
        with open(os.path.join(json_dir, json_file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        data = merge_words_by_speaker(data)

        text_features = []
        video_features = []
        audio_features = []

        speaker_num = []

        offset = 0  # 노드 인덱스 오프셋 초기화

        for item in tqdm(data, desc="Processing utterances"):
            word = item['word']
            speaker = item['speaker']
            start = item['start']
            end = item['end']

            if not word or (end - start) < 1:
                offset += 1
                continue

            speaker_num.append(speaker)

            # 텍스트 임베딩
            with open(os.path.join(data_dir, "text_features", json_file.replace(".json", f"/{speaker}_{start:.2f}_{end:.2f}.pt")), 'rb') as f:
                text_feature:torch.Tensor = torch.load(f)
                text_features.append(text_feature)

            # 비디오 임베딩
            with open(os.path.join(data_dir, "video_features", json_file.replace(".json", f"/{speaker}_{start:.2f}_{end:.2f}.pt")), 'rb') as f:
                video_feature = torch.load(f)
                video_features.append(video_feature)

            # 오디오 임베딩
            with open(os.path.join(data_dir, "audio_features", json_file.replace(".json", f"/{speaker}_{start:.2f}_{end:.2f}.pt")), 'rb') as f:
                audio_feature = torch.load(f)
                audio_features.append(audio_feature)

        # 대화 그래프에 노드 추가
        conversation_graph.x_text = torch.stack(text_features,dim=0)
        conversation_graph.x_video = torch.stack(video_features,dim=0)
        conversation_graph.x_audio = torch.stack(audio_features,dim=0)
        conversation_graph.speaker_num = torch.tensor(speaker_num, dtype=torch.long) if speaker_num else None
        conversation_graph.global_index = []
        conversation_graph.speaker_index = []

        node_count = len(text_features)  # 노드 개수
        
        speaker_color = [
    'blue',      # 1
    'green',     # 2
    'orange',    # 3
    'skyblue',   # 4
    'cyan',      # 5
    'gold',      # 6
    'lime',      # 7
    'brown',     # 8
    'pink',      # 9
    'gray'       # 10
]

        node_color = []

        # 대화 그래프에 Sequential 엣지 추가
        for idx in range(node_count):
            node_color.append('red')
            if idx == 0:
                continue
            # 이전 노드와 현재 노드 간의 엣지 추가
            elif idx == 1:
                conversation_graph.edge_index = torch.tensor([[idx - 1, idx], [idx, idx - 1]], dtype=torch.long)
            else:
                conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[idx - 1, idx], [idx, idx - 1]], dtype=torch.long)], dim=1)


        # 대화 그래프에 Hierarchical-conversation 노드 임베딩 초기화
        conversation_graph.x_hie_conv = torch.zeros((1, 768), dtype=torch.float)  # Hierarchical-conversation 노드 임베딩 초기화 (추후 nn.Parameter로 교체 가능)

        conversation_graph.global_index.append(node_count)
        node_color.append('purple')

        for i in range(node_count):
            conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[i], [node_count]], dtype=torch.long)], dim=1)
            conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[node_count], [i]], dtype=torch.long)], dim=1)

        prev_speaker_num = 0
        unique_speakers = set(speaker_num)
        conversation_graph.x_hie_speaker = torch.zeros((len(unique_speakers), 768), dtype=torch.float)  # Hierarchical-speaker 노드 임베딩 초기화
        conversation_graph.speaker_index.extend([node_count+1+i] for i in range(len(unique_speakers)))  # Hierarchical-speaker 노드 인덱스 추가

        for i, speaker in enumerate(unique_speakers):
            speaker_indices = [i for i, s in enumerate(speaker_num) if s == speaker]
            node_color.append(speaker_color[speaker])

            for s in speaker_indices:
                conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[s], [node_count+1+i]], dtype=torch.long)], dim=1)
                conversation_graph.edge_index = torch.cat([conversation_graph.edge_index, torch.tensor([[node_count+1+i], [s]], dtype=torch.long)], dim=1)

        if draw:
            G = nx.DiGraph()

            for edge in conversation_graph.edge_index.t().tolist():
                G.add_edge(edge[0], edge[1])

            # pos = nx.spring_layout(G)  # 노드 위치 설정
            net = Network(notebook=False, height='900px', width='100%')
            net.from_nx(G)

            for node in net.nodes:
                node["label"] = str(node["id"])  # id를 문자열로 label로 사용
                node["color"] = node_color[node["id"]]  # 기존 색상 그대로
                
            net.show('graph.html',notebook=False)

        base = json_file.replace(".json", "")

        output_path = os.path.join(output_dir,f"{base}_single.pt")
        torch.save(conversation_graph,output_path)
    
if __name__ == "__main__":
    
    json_dir = "data/orch"
    video_dir = "data/video"
    audio_dir = "data/audio"
    feature_dir = "data/processed"
    graph_dir = "data/graph"
    
    # json_to_multi_modal_embedding("data/orch","data/video","data/audio","data/processed")
    
    construct_conversation_graph(json_dir,feature_dir,graph_dir,True)

    construct_conversation_graph_with_single_global(json_dir,feature_dir,graph_dir)