from dataset.droplets import DropletDataset
from torch.utils.data import DataLoader
from model.dropletnet import DropletNet
from tqdm import tqdm
import torch
import sys
import os
import wandb
import numpy as np

wandb.init(project="ddp-training-all_classes-infocus", entity="unnikrishnan")
EPOCHS = int(sys.argv[1])
directory = "train_" + sys.argv[2]
parent_dir = "model"
train_folder_path = os.path.join(parent_dir, directory)
print(os.getcwd())
print(os.listdir(os.getcwd()))
os.mkdir(train_folder_path)

ROOT_DIR = "data"
TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
TEST_CSV = os.path.join(ROOT_DIR, "test.csv")
VALID_CSV = os.path.join(ROOT_DIR, "valid.csv")


train_dataset = DropletDataset(root_dir = ROOT_DIR, annotations_path = TRAIN_CSV)
test_dataset = DropletDataset(root_dir = ROOT_DIR, annotations_path = TEST_CSV)
valid_dataset = DropletDataset(root_dir = ROOT_DIR, annotations_path = VALID_CSV)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img, label = next(iter(train_dataloader))
model = DropletNet()
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
criterion = torch.nn.MSELoss()

wandb.config = {
  "learning_rate": 0.001,
  "epochs": EPOCHS,
  "batch_size": 32
}
train_loss = []
avg_train_loss = np.inf
val_loss = []
avg_val_loss = np.inf
for epoch in range(1, EPOCHS+1):
    model.train()
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            preds = model(data).squeeze()
            outputs = torch.sigmoid(preds)
            loss = criterion(target.float(), outputs)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if epoch != 1:
                avg_train_loss = np.mean(train_loss)
            wandb.log({"loss": loss, "avg_loss": avg_train_loss}) 
            tepoch.set_postfix(mse = loss.item(), avg_mse = avg_train_loss)
    model.eval()   
    with tqdm(valid_dataloader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Validation {epoch}")
            data, target = data.to(device), target.to(device)
           
            preds = model(data).squeeze()
            outputs = torch.sigmoid(preds)
            loss = criterion(target.float(), outputs)
            val_loss.append(loss.item())
            if epoch != 1:
                avg_val_loss = np.mean(val_loss)
            wandb.log({"validation_loss": loss, "avg_validation_loss": avg_val_loss})
            tepoch.set_postfix(mse = loss.item(), avg_mse = avg_val_loss)
    if epoch%5 == 0:
        tqdm.write("Saving model_" + sys.argv[2] + str(epoch) + ".torch")
        torch.save(model.state_dict(), os.path.join(train_folder_path, 'model' + str(epoch) + ".torch"))