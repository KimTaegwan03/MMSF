import numpy as np
import torch
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from torch_geometric.loader import DataLoader
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


# 1) 데이터 로드
from dataset import GraphDataset  # 유저 정의 모듈
annotation_path = "totalLabel.csv"
graph_dir       = "/mnt/share65/emb/graph"
dataset = GraphDataset(annotation_path, graph_dir)
loader  = DataLoader(dataset, batch_size=1, shuffle=False)

# 2) 임베딩 및 타깃 추출
embs, targets = [], []
for batch in loader:
    batch = batch.to('cpu')  # 혹은 device
    feat = batch.x['sentence']            # [node_num, 768]
    emb  = feat.mean(dim=0).numpy()       # mean pooling → [768]
    embs.append(emb)
    targets.append(batch.y.cpu().numpy().ravel())  # [1]

X = np.stack(embs, axis=0).squeeze()     # [N, 768]
y = np.vstack(targets)         # [N, 1]

# 3) 표준화
Xs = StandardScaler().fit_transform(X)
ys = StandardScaler().fit_transform(y).ravel()

# 4) PLS 회귀 모델 학습 및 latent score 변환
n_comp = min(10, Xs.shape[1])
pls    = PLSRegression(n_components=n_comp)
pls.fit(Xs, ys)
X_scores, Y_scores = pls.transform(Xs, ys)  # both [N, n_comp]

# 5) 1st component 상관계수
r, pval = pearsonr(X_scores[:, 0], Y_scores[:, 0])
print(f"PLS Pearson r (1st component): {r:.4f} (p={pval:.2e})")

# 6) 교차검증 Q² 계산
cv = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(pls, Xs, ys, cv=cv)
Q2 = 1 - np.sum((ys - y_pred_cv) ** 2) / np.sum((ys - ys.mean()) ** 2)
print(f"PLS cross-validated Q²: {Q2:.4f}")


def evaluate_regressor(model, X, y, cv):
    """교차검증 예측 후 Pearson r, Q² 계산"""
    # cross_val_predict: 각 fold의 test set에 대해 예측
    y_pred = cross_val_predict(model, X, y, cv=cv)
    # Pearson 상관계수
    r, pval = pearsonr(y_pred, y)
    # Q² (cross-validated R²)
    Q2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    return r, pval, Q2

# --- 1) SVR (RBF 커널) ---
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
r_svr, p_svr, Q2_svr = evaluate_regressor(svr, Xs, ys, cv)
print(f"SVR RBF    → Pearson r: {r_svr:.3f} (p={p_svr:.2e}), Q²: {Q2_svr:.3f}")

# --- 2) Gradient Boosting Regressor ---
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
r_gbr, p_gbr, Q2_gbr = evaluate_regressor(gbr, Xs, ys, cv)
print(f"GBR        → Pearson r: {r_gbr:.3f} (p={p_gbr:.2e}), Q²: {Q2_gbr:.3f}")

# --- 3) MLP Regressor ---
mlp = MLPRegressor(hidden_layer_sizes=(256,128), activation='relu',
                   solver='adam', alpha=1e-4, max_iter=200, random_state=42)
r_mlp, p_mlp, Q2_mlp = evaluate_regressor(mlp, Xs, ys, cv)
print(f"MLP        → Pearson r: {r_mlp:.3f} (p={p_mlp:.2e}), Q²: {Q2_mlp:.3f}")