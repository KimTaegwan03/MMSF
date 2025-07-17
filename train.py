from VideoUniGraph import VideoUniGraph
from data.datamodule import MultimodalDataset

dataset = MultimodalDataset("data/embedding","train","cuda")
model = VideoUniGraph(
    {'text':768}
).to('cuda')

data = dataset.get(1)

print(data.shape)

output = model({'text':data}, return_embeddings=True)

print(output)
