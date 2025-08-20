import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import os

# 1. 커스텀 Dataset 클래스 정의
class GraphDataset(Dataset):
    def __init__(self, label_annotation_path, graph_data_dir):
        """
        data_list: List[torch_geometric.data.Data]
        """
        annotations = pd.read_csv(label_annotation_path,index_col=[0],header=[0]).reset_index(drop=True)

        self.data_list = []
        for i in range(len(annotations)):
            video_id = annotations['video_id'][i]
            name = annotations['name'][i]
            label = annotations['label'][i]
            # label = annotations['total_label'][i]
            # cls_label = annotations['cls_label'][i]
            # or_label = annotations['or_label'][i]

            graph_path = os.path.join(graph_data_dir, f"{video_id}.pt")
            if not os.path.exists(graph_path):
                continue  # 또는 raise Exception 등 처리

            data = torch.load(graph_path)  # Should be a torch_geometric.data.Data object
            data.name = name
            data.y = torch.tensor([label], dtype=torch.float16)  # Target 값 추가
            # data.y_cls = torch.tensor([cls_label], dtype=torch.float16)  # Target 값 추가
            # data.y_or = torch.tensor([or_label], dtype=torch.float16)  # Target 값 추가
            # data.num_nodes = len(data.x) + data.speaker_num + 1
            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    

if __name__ == "__main__":
    dataset = GraphDataset("person_label.csv","/mnt/share65/emb/graph/flat_word")

    print(dataset.data_list)