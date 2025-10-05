from dataset.camvid import CamVidDataset
from torch.utils.data.dataloader import DataLoader
import torch
val_dataset = CamVidDataset("data/raw/CamVid/val","data/raw/CamVid/val_labels")
val_dataloader = DataLoader(batch_size=8,dataset=val_dataset,pin_memory=True,num_workers=4)

def validate_(model,criterion):
    model.eval()
    val_loss = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for images,labels in val_dataloader:
            images,labels = images.to(device),labels.to(device)
            output = model(images)
            loss = criterion(output,labels)
            val_loss+=loss.item()
    val_loss/=len(val_dataloader)
    print(f"val_loss is {val_loss}")
    return val_loss