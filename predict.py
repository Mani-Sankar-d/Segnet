import torch
import matplotlib.pyplot as plt
from models.segnet import Segnet
from models.blocks import Encoder, Decoder
from dataset.camvid import CamVidDataset
from torch.utils.data import DataLoader
from utils.visualize import visualize_prediction

n_classes = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CamVidDataset("data/raw/CamVid/test", "data/raw/CamVid/test_labels")
loader = DataLoader(dataset, batch_size=4, shuffle=False)

encoder = Encoder()
decoder = Decoder(n_classes)
model = Segnet(encoder, decoder).to(device)


checkpoint = torch.load("best_checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


images, labels = next(iter(loader))
images = images.to(device)

with torch.no_grad():
    for images,labels in loader:
        images = images.to(device)
        outputs = model(images) 
        preds = torch.argmax(outputs, dim=1).cpu()  # [B, H, W]
        for i in range(4):
            visualize_prediction(images[i], preds[i],labels[i])