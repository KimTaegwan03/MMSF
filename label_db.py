import sqlite3
import pandas as pd
import os
import json

db_path = "/mnt/share65/speech_segments2.db"

graph_dir = "/mnt/share65/emb/graph"
folder_path = "/mnt/share65/asd_orch_jsons"

# 결과 저장용 딕셔너리
unknown_ratio_per_file = {}
non_speaker_names_per_file = {}  # 파일별 "SPEAKER"가 포함되지 않은 speaker name들

# 폴더 내 모든 JSON 파일 순회
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        
        # JSON 데이터 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 전체 단어 개수
        total_count = len(data)
        
        # speaker가 'unknown'인 단어 개수
        unknown_count = sum(
            1 for item in data if "SPEAKER" in str(item.get("speaker", ""))
        )
        
        # 비율 계산
        ratio = unknown_count / total_count if total_count > 0 else 0
        
        # 결과 딕셔너리에 저장
        unknown_ratio_per_file[filename] = ratio
        
        # "SPEAKER"가 포함되지 않은 unique speaker names 추출
        non_speaker_names = set()
        for item in data:
            speaker = str(item.get("speaker", ""))
            if "SPEAKER" not in speaker and speaker:  # "SPEAKER"가 포함되지 않고 빈 문자열이 아닌 경우
                non_speaker_names.add(speaker)
        
        non_speaker_names_per_file[filename] = non_speaker_names

# non_SPEAKER 비율이 0.5 이상인 파일들의 speaker names만 추출
high_non_speaker_files = {
    filename.replace(".json",""): names 
    for filename, names in non_speaker_names_per_file.items() 
    if (1 - unknown_ratio_per_file[filename]) >= 0.5  # non_SPEAKER 비율이 0.5 이상
}

file_list = sorted(list(high_non_speaker_files.keys()))

conn = sqlite3.connect(db_path)

df = pd.read_sql_query("SELECT video_id, persons_found as label FROM video_metadata ORDER BY video_id", conn)


# df['video_id_normalized'] = df['video_id'].str.replace('_', '-', regex=False)

filtered_df = df[df['video_id'].isin(file_list)]

print(len(filtered_df))

labels = filtered_df['label']

new_data = []

tot = []
cls_tot = []
or_label = []

for idx, row in filtered_df.iterrows():
    label = json.loads(row['label'])
    video_id = row['video_id']
    
    if video_id in high_non_speaker_files:
        for n in high_non_speaker_files[video_id]:
            if n.replace('_',' ') in label.keys():
                new_data.append(
                    {
                        "video_id": video_id,
                        "name": n.replace('_',' '),
                        "label": label[n.replace('_',' ')]
                    }
                )
    # tot.append(sum(label.values()) / len(label))
    # cls_tot.append(0 if sum(label.values()) / len(label) < 0.3 else 1)
    # or_label.append(1 if sum(label.values()) > 0.0 else 0)

new_df = pd.DataFrame(new_data)

new_df.to_csv("person_label.csv")

print(new_df['label'].mean())

exit()

filtered_df['total_label'] = tot
filtered_df['cls_label'] = cls_tot
filtered_df['or_label'] = or_label

filtered_df.to_csv("totalLabel.csv")

print(sum(cls_tot)/len(cls_tot))
print(sum(or_label)/len(or_label))