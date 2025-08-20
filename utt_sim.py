import os
import torch
from sklearn.metrics.pairwise import cosine_similarity

json_dir = "/mnt/share65/orch_jsons"
video_dir = "/mnt/share65/videos"
# audio_dir = "data/audio"
feature_dir = "/mnt/share65/emb"
graph_dir = "/mnt/share65/emb/graph"

text_emb_dir = feature_dir + "/sentence"

# dir_list = os.listdir(text_emb_dir)
# dir_list.sort()

# for video_id in dir_list:
#     feat_list = os.listdir(os.path.join(text_emb_dir,video_id))

#     pt_list = []

#     for filename in feat_list:
#         filepath = text_emb_dir + "/" + video_id + "/" + filename
#         if filepath.endswith("pt"):
#             pt_list.append(torch.load(text_emb_dir + "/" + video_id + "/" + filename))

#     for i in range(len(feat_list)-1):
#         text_i = pt_list[i]

#         for j in range(i+1,len(feat_list)-1):
#             text_j = pt_list[j]

#             print(f"Similarity {i} and {j}:", cosine_similarity(text_i,text_j))


dir_list = os.listdir(graph_dir)
dir_list.sort()

for graph_file in dir_list:
    if graph_file.endswith("_flat.pt"):
        graph = torch.load(graph_dir+"/"+graph_file)
        graph_x = graph.x['sentence']
        
        for i in range(len(graph_x)):
            text_i = graph_x[i]

            for j in range(i+1,len(graph_x)):
                text_j = graph_x[j]

                print(f"Similarity {i} and {j}:", cosine_similarity(text_i,text_j))

    

