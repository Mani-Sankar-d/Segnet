import torch
import torch.nn as nn
from models.segnet import Segnet
from models.blocks import Encoder, Decoder
from dataset.camvid import CamVidDataset
from torch.utils.data.dataloader import DataLoader
from utils.checkpoint import save_checkpoint,load_checkpoint
from validate import validate_
import os
import numpy as np
def train(total_epochs, start_epoch=0):
    print("Traning starts ")
    global best_val_loss,counter,patience
    for epoch in range(start_epoch, total_epochs):
        epoch_loss=0
        model.train()
        for images,labels in dataloader:
            images,labels = images.to(device,non_blocking=True),labels.to(device,non_blocking=True)
            output = model(images)
            loss = criterion(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
        print(f"loss for epoch {epoch+1} is {epoch_loss}")
        val_loss = validate_(model,criterion)
        scheduler.step()
        save_checkpoint(model,optimizer,scheduler,epoch,val_loss,counter,"last_checkpoint.pth")
        
        if val_loss<best_val_loss:
            best_val_loss=val_loss
            counter = 0
            save_checkpoint(model,optimizer,scheduler,epoch,val_loss,counter,"best_checkpoint.pth")
        else:
            counter += 1
            print(f"Epoch {epoch}: No improvement ({counter}/{patience})")
        
        if counter >= patience:
            print("Early stopping triggered!")
            break

if __name__=="__main__":
    n_classes = 32
    lr = 0.01
    momentum = 0.9
    encoder = Encoder()
    decoder = Decoder(n_classes)
    model = Segnet(encoder=encoder,decoder=decoder)
    optimizer = torch.optim.SGD(params=model.parameters(),lr=lr,momentum=momentum,weight_decay=0.0005)
    weights = np.load("utils/weights.npy")
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights,dtype=torch.float32).cuda()
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=30,gamma=0.1)
    dataset = CamVidDataset("data/raw/CamVid/train","data/raw/CamVid/train_labels")
    dataloader = DataLoader(dataset=dataset,batch_size=8,shuffle=True,num_workers=4,pin_memory=True)
    best_val_loss = float('inf')
    counter=0
    patience = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model,optimizer,scheduler,start_epoch,best_val_loss,counter = load_checkpoint(model,optimizer,scheduler,"last_checkpoint.pth",device)
    train(100,start_epoch)