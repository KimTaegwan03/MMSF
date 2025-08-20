import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from dataset import GraphDataset
from VideoUniGraph import VideoUniGraph
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from torch.utils.data import random_split

# 설정값 (train()와 동일하게 맞춰주세요)
annotation_path = "person_label.csv"
graph_dir       = "/mnt/share65/emb/graph/log"
save_path       = "results/last_cls_model.pt"
device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) 전체 데이터셋 로드
dataset = GraphDataset(annotation_path, graph_dir)
# loader  = DataLoader(dataset, batch_size=1, shuffle=False)

n = len(dataset)
train_size = int(0.8 * n)
val_size = int(0.1 * n)
test_size = n - train_size - val_size
split_gen = torch.Generator().manual_seed(42)

train_dataset, _, test_dataset = random_split(dataset, [train_size, val_size, test_size],split_gen)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 2) 모델 생성 및 저장된 state 불러오기
model = VideoUniGraph(
    {'sentence': 768},
    hidden_dim=256,
    num_layers=2,
    dtype=torch.float32
).to(device)
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()

# 3) 예측값·실제값 수집
preds, targets = [], []
with torch.no_grad():
    model.eval()
    for batch in test_loader:
        batch = batch.to(device)
        out:torch.Tensor = model(features=batch.x, graph=batch, decoding=True, all=False)[batch.name[0]].float().view(-1)
        preds.append(torch.tensor([1]) if out.cpu() > 0.5 else torch.tensor([0]))
        targets.append(batch.y.cpu())

# 4) 넘파이 배열로 변환
y_true = torch.tensor(targets).numpy()
y_pred = torch.tensor(preds).numpy()

cm = confusion_matrix(y_true, y_pred)

# === 방법 B: seaborn heatmap ===
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=[0,1], yticklabels=[0,1], ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix (Seaborn)")
plt.show()