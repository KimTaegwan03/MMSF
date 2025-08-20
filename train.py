from VideoUniGraph import VideoUniGraph
from dataset import GraphDataset
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm
from torch.utils.data import random_split, WeightedRandomSampler
import torch.nn as nn
import wandb
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

annotation_path = "person_label.csv"
graph_dir = "/mnt/share65/emb/graph/flat_word"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from sklearn.metrics import accuracy_score,mean_squared_log_error, f1_score, recall_score, precision_score

lr = 5e-5
epochs = 100
lambda_var = 2e-2
min_var = 0.24  # 데이터와 배치 크기에 맞춰 조정
batch_steps = 32
weight_decay = 1e-4
hidden_dim = 256
num_layers = 2
save_path = "results/best_model.pt"

# 초기화
wandb.init(project=f"video-trends-model-spk", config={
    "learning_rate": lr,
    "weight_decay": weight_decay,
    "epochs": epochs,
    "batch_size": 1,
    "lambda_var": lambda_var,
    "min_var": min_var,
    "batch_steps": batch_steps,
    "hidden_dim": hidden_dim,
    "num_layers":num_layers,
    "model": "VideoUniGraph"
})


@torch.no_grad()
def evaluate_cls(model, val_loader, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    for batch in val_loader:
        batch = batch.to(device)
        output = model(features=batch.x, graph=batch, decoding=True, all=False)[batch.name[0]].float()
        preds = (torch.sigmoid(output) >= threshold).long().cpu().numpy()
        labels = batch.y.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
    acc = accuracy_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    pre = precision_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, rec, pre, f1

def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(features=batch.x, graph=batch, decoding=True).float()
            target = batch.y.to(device).float()
            loss = loss_fn(output.squeeze(0), target)
            total_loss += loss.item()
    return total_loss / len(loader)

def train_cls(model, loader, val_loader, optim, loss_fn, epochs, verbose):
    best_val_acc = -1
    save_path = "results/best_cls_model.pt"
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0.0
        pred_buffer = []
        target_buffer = []

        for step, batch in enumerate(loader):
            batch = batch.to(device)
            output = model(features=batch.x, graph=batch, decoding=True).float().view(-1)
            target = batch.y_or.float().view(-1)
            # loss = loss_fn(output, target)
            pred_buffer.append(output)
            target_buffer.append(target)

            if step % batch_steps == 0 or step == len(loader):
                # concatenate
                preds   = torch.cat(pred_buffer)   # shape [accum_steps]
                targets = torch.cat(target_buffer) # shape [accum_steps]

                # 4) 기본 손실
                loss_bce = F.binary_cross_entropy_with_logits(preds, targets,weight=torch.tensor(0.6307))

                # 5) 분산 페널티
                var_pred   = torch.var(preds, unbiased=False)
                var_penalty = (min_var - var_pred)**2   # 분산이 min_var 밑으로 내려가면 양수
                # (원하면 **2를 붙여 페널티를 더 강하게 할 수도 있습니다)

                # 6) 최종 손실
                loss = loss_bce + lambda_var * var_penalty

                # print(loss.item())

                # wandb 로깅
                wandb.log({"train_loss_step": loss.item(),
                            "var_pred": var_pred})
                
                optim.zero_grad()
                loss.backward()
                optim.step()

                # 8) 버퍼 비우기
                pred_buffer.clear()
                target_buffer.clear()

                total_loss += loss.item()

        if verbose:
            train_acc, train_rec, train_pre, train_f1 = evaluate_cls(model, loader)
            val_acc, val_rec, val_pre, val_f1 = evaluate_cls(model, val_loader)
            # avg_loss = total_loss / len(loader) * batch_steps
            # 최고 성능 경신 시 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)
            # 에포크 단위 로깅
            wandb.log({
                "epoch": epoch,
                # "train_loss": avg_loss,
                "train_acc": train_acc,
                "val. acc.": val_acc,
                "valid. recall": val_rec,
                "valid. precision": val_pre,
                "valid. F1": val_f1
            })

def train_spk(model, loader, val_loader, optim, loss_fn, epochs, verbose):
    best_val_acc = -1
    save_path = "results/best_cls_model.pt"
    last_path = "results/last_cls_model.pt"
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0.0
        pred_buffer = []
        target_buffer = []

        for step, batch in enumerate(loader):
            batch = batch.to(device)
            output = model(features=batch.x, graph=batch, decoding=True, all=False)[batch.name[0]].float().view(-1)
            target = batch.y.float().view(-1)
            # loss = loss_fn(output, target)
            pred_buffer.append(output)
            target_buffer.append(target)

            if step % batch_steps == 0 or step == len(loader):
                # concatenate
                preds   = torch.cat(pred_buffer)   # shape [accum_steps]
                targets = torch.cat(target_buffer) # shape [accum_steps]

                # 4) 기본 손실
                loss = loss_fn(preds, targets)

                # 5) 분산 페널티
                var_pred   = torch.var(preds, unbiased=False)
                var_penalty = (min_var - var_pred)**2   # 분산이 min_var 밑으로 내려가면 양수
                # (원하면 **2를 붙여 페널티를 더 강하게 할 수도 있습니다)

                # 6) 최종 손실
                loss = loss # + lambda_var * var_penalty

                # print(loss.item())

                # wandb 로깅
                wandb.log({"train_loss_step": loss.item(),
                            "var_pred": var_pred})
                
                optim.zero_grad()
                loss.backward()
                optim.step()

                # 8) 버퍼 비우기
                pred_buffer.clear()
                target_buffer.clear()

                total_loss += loss.item()

        if verbose:
            train_acc, train_rec, train_pre, train_f1 = evaluate_cls(model, loader)
            val_acc, val_rec, val_pre, val_f1 = evaluate_cls(model, val_loader)
            # avg_loss = total_loss / len(loader) * batch_steps
            # 최고 성능 경신 시 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)
            torch.save(model.state_dict(), last_path)
            # 에포크 단위 로깅
            wandb.log({
                "epoch": epoch,
                # "train_loss": avg_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "valid. recall": val_rec,
                "valid. precision": val_pre,
                "valid. F1": val_f1
            })

def train(model, loader, val_loader, optim, loss_fn, epochs, verbose):
    best_val_loss = np.inf
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0.0
        pred_buffer = []
        target_buffer = []

        for step,batch in enumerate(loader):
            batch = batch.to(device)
            output = model(features=batch.x, graph=batch, decoding=True).float().view(-1)
            target = batch.y.float().view(-1)
            # loss = loss_fn(target, output)
            pred_buffer.append(output)
            target_buffer.append(target)

            if step % batch_steps == 0 or step == len(loader):
                # concatenate
                preds   = torch.cat(pred_buffer)   # shape [accum_steps]
                targets = torch.cat(target_buffer) # shape [accum_steps]

                # 4) 기본 MSE 손실
                loss_mse = F.mse_loss(preds, targets)

                # 5) 분산 페널티
                var_pred   = torch.var(preds, unbiased=False)
                var_penalty = (min_var - var_pred)   # 분산이 min_var 밑으로 내려가면 양수
                # (원하면 **2를 붙여 페널티를 더 강하게 할 수도 있습니다)

                # 6) 최종 손실
                loss = loss_mse + lambda_var * var_penalty

                # wandb 로깅
                wandb.log({"train_loss_step": loss.item(),
                            "var_pred": var_pred})

                # 7) 업데이트
                optim.zero_grad()
                loss.backward()
                optim.step()

                # 8) 버퍼 비우기
                pred_buffer.clear()
                target_buffer.clear()

                total_loss += loss.item()

            if not torch.isfinite(loss):
                print(loss)
                print("Loss is NaN or Inf!")


        if verbose:
            avg_val_loss = evaluate(model, val_loader, loss_fn)
            avg_loss = total_loss / len(loader) * batch_steps

            # 최고 성능 경신 시 모델 저장
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)

            # 에포크 단위 로깅
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
                # "pred_mean": np.mean(outputs).item(),
                # "pred_std": np.std(outputs).item()
            })

import matplotlib.pyplot as plt
import torch

def plot_predictions(model, loader, device):
    model.eval()
    preds = []
    targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(features=batch.x, graph=batch, decoding=True).float()
            target = batch.y.to(device).float()

            preds.append(output.cpu())
            targets.append(target.cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    # --- 1. 예측값 분포 ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(preds, bins=30, alpha=0.7, label='Predicted')
    plt.hist(targets, bins=30, alpha=0.7, label='Ground Truth')
    plt.title("Prediction vs Ground Truth Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()

    # --- 2. 산점도 ---
    plt.subplot(1, 2, 2)
    plt.scatter(targets, preds, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # 이상적 직선
    plt.title("Predicted vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    import numpy as np

    dataset = GraphDataset(annotation_path,graph_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)


    n = len(dataset)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    test_size = n - train_size - val_size
    split_gen = torch.Generator().manual_seed(42)

    train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, test_size],split_gen)

    # 각 샘플별 라벨 추출 (train_dataset은 Subset이므로 .dataset.data_list에서 접근)
    labels = torch.tensor([train_dataset[i].y.item() for i in range(len(train_dataset))], dtype=torch.long)

    # 클래스별 개수 계산
    class_sample_count = torch.bincount(labels)  # tensor([neg_count, pos_count])
    # 각 클래스에 대한 가중치 (빈도수가 적을수록 가중치 높게)
    weights = 1.0 / class_sample_count.float()

    # 샘플별 weight 매핑
    sample_weights = weights[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),  # 한 epoch에 뽑을 샘플 수
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


    model = VideoUniGraph(
        {'word':768},
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        feat_noise=0.01,
        dtype = torch.float32
    ).to('cuda')

    params_to_optimize = list(model.modal_projector.parameters()) + \
                     list(model.gnn_layers.parameters()) + \
                     list(model.speaker_decoder.parameters()) + \
                     list(model.transformer.parameters()) + \
                     list(model.pos_emb.parameters()) + \
                     list(model.entire_decoder.parameters()) + [
                         model.mask_token,
                         model.x_hie_conv_token,
                         model.x_hie_speaker_token
                     ]

    # 옵티마이저와 손실 함수 정의
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=weight_decay)

#     scheduler = ReduceLROnPlateau(
#     optimizer,
#     mode='max',
#     factor=0.5,
#     patience=3,
#     verbose=True
# )
    
    loss_fn = nn.MSELoss()
    # loss_fn = mean_squared_log_error
    loss_fn_cls = nn.BCEWithLogitsLoss()

    # train(model, train_loader, val_loader, optimizer, loss_fn, epochs=epochs, verbose=True)
    train_spk(model, train_loader, val_loader, optimizer, loss_fn_cls, epochs=epochs, verbose=True)

    


# data = dataset.get(1)

# print(data.shape)

# output = model(data.x, return_embeddings=True)

# print(output)
