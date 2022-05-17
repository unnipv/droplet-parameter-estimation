from dataset.droplets import DropletDataset
from torch.utils.data import DataLoader
from model.Unninet import UnniNet
from tqdm import tqdm
import torch
import sys
import os
import wandb
from pathlib import Path

wandb.init(project="ddp-training", entity="unnikrishnan")
EPOCHS = int(sys.argv[1])
directory = "train_" + sys.argv[2]
parent_dir = "model"
train_folder_path = os.path.join(parent_dir, directory)
print(os.getcwd())
print(os.listdir(os.getcwd()))
os.mkdir(train_folder_path)

ROOT_DIR = "data\m_crop"
TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
TEST_CSV = os.path.join(ROOT_DIR, "test.csv")
VALID_CSV = os.path.join(ROOT_DIR, "valid.csv")


train_dataset = DropletDataset(root_dir = ROOT_DIR, annotations_path = TRAIN_CSV)
test_dataset = DropletDataset(root_dir = ROOT_DIR, annotations_path = TEST_CSV)
valid_dataset = DropletDataset(root_dir = ROOT_DIR, annotations_path = VALID_CSV)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img, label = next(iter(train_dataloader))
model = UnniNet()
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
criterion = torch.nn.MSELoss()

wandb.config = {
  "learning_rate": 0.0001,
  "epochs": EPOCHS,
  "batch_size": 64
}

for epoch in range(1, EPOCHS+1):
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
            wandb.log({"loss": loss})
            tepoch.set_postfix(mse=loss.item())
    
    with tqdm(valid_dataloader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Validation {epoch}")
            data, target = data.to(device), target.to(device)
           
            preds = model(data).squeeze()
            outputs = torch.sigmoid(preds)
            loss = criterion(target.float(), outputs)
            wandb.log({"validation_loss": loss})
            tepoch.set_postfix(mse = loss.item())
    if epoch%50 == 0:
        tqdm.write("Saving model_" + sys.argv[2] + str(epoch) + ".torch")
        torch.save(model.state_dict(), os.path.join(train_folder_path, 'model' + str(epoch) + ".torch"))