import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv('person_label.csv')  # 실제 CSV 경로로 변경

# name별 개수 계산
name_counts = df['name'].value_counts()

# 출력
print(name_counts)

print(len(df['name'].unique()))
