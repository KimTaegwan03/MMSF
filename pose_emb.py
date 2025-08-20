from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model

# Predict
results = model("https://ultralytics.com/images/bus.jpg")

# Get keypoints
for result in results:
    keypoints = result.keypoints.xy  # (n, k, 2) where n: number of people, k: keypoints, 2: (x, y)
    
    for person_keypoints in keypoints:
        # Flatten the keypoints into a 1D vector (x1, y1, x2, y2, ..., xk, yk)
        pose_embedding = person_keypoints.cpu().numpy()
        print("Pose embedding shape:", pose_embedding.shape)
        print("Pose embedding vector:", pose_embedding)