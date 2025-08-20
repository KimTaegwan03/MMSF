import sqlite3
import pandas as pd
import os
import json

db_path = "/mnt/share65/speech_segments.db"

graph_dir = "/mnt/share65/emb/graph"

graph_files_list = os.listdir(graph_dir)
graph_files_list.sort()

print(len(graph_files_list))

graph_base_list = [os.path.splitext(f)[0].replace('_flat','') for f in graph_files_list]

conn = sqlite3.connect(db_path)

df = pd.read_sql_query("SELECT video_id, persons_found as label FROM video_metadata WHERE channel_name = 'Jimmy Kimmel Live'", conn)

print(df.head(5))
print(df.columns)

# # df['video_id_normalized'] = df['video_id'].str.replace('_', '-', regex=False)

# filtered_df = df[df['video_id'].isin(graph_base_list)]

# labels = filtered_df['label']

# tot = []
# cls_tot = []
# or_label = []

# for l in labels:
#     label = json.loads(l)

#     tot.append(sum(label.values()) / len(label))
#     cls_tot.append(0 if sum(label.values()) / len(label) < 0.3 else 1)
#     or_label.append(1 if sum(label.values()) > 0.0 else 0)

# filtered_df['total_label'] = tot
# filtered_df['cls_label'] = cls_tot
# filtered_df['or_label'] = or_label

# filtered_df.to_csv("totalLabel.csv")

# print(sum(cls_tot)/len(cls_tot))
# print(sum(or_label)/len(or_label))