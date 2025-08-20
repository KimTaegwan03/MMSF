import os
import torch
from PIL import Image
from torchvision import transforms
from sklearn.cluster import KMeans
from tqdm import tqdm
import shutil
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import DBSCAN
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4815, 0.4578, 0.4082),
                         std=(0.2686, 0.2613, 0.2758)),
])

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().cpu().numpy()

def cluster_with_dbscan(embeddings, eps=0.3, min_samples=3):
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = clustering.fit_predict(embeddings)
    return labels

def cluster_and_filter_images(folder_path, n_clusters=2, keep_cluster=None):
    image_paths = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.png'))
    ])

    if len(image_paths) < n_clusters:
        print(f"[SKIP] Not enough images to cluster in {folder_path}")
        return

    print(f"[INFO] Processing {folder_path} with {len(image_paths)} images")

    embeddings = []
    for path in tqdm(image_paths, desc=f"Embedding"):
        emb = get_image_embedding(path)
        embeddings.append(emb)

    # # 3. DBSCAN clustering 
    # labels = cluster_with_dbscan(embeddings)      # Worse than K-Means clustering

    # 3. KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # 4. Identify main cluster (with most elements)
    counts = {label: list(labels).count(label) for label in set(labels)}
    main_cluster = max(counts, key=counts.get) if keep_cluster is None else keep_cluster

    # 5. Keep only main cluster images
    filtered_dir = folder_path + "_filtered"
    os.makedirs(filtered_dir, exist_ok=True)

    for path, label in zip(image_paths, labels):
        if label == main_cluster:
            shutil.copy(path, os.path.join(filtered_dir, os.path.basename(path)))

    print(f"[DONE] Kept {counts[main_cluster]} / {len(image_paths)} images in '{filtered_dir}'")

# Example usage:
# 스피커 폴더 반복
base_dir = "spekaer_img"  # 예: speaker_07, speaker_01 ... 이런 폴더들이 들어있는 디렉토리
for speaker_folder in os.listdir(base_dir):
    speaker_path = os.path.join(base_dir, speaker_folder)
    if os.path.isdir(speaker_path):
        cluster_and_filter_images(speaker_path)
