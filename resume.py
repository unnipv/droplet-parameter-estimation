from dataset.droplets import DropletDataset
from torch.utils.data import DataLoader
from model.Unninet import UnniNet
from tqdm import tqdm
import torch
import sys

ROOT_DIR = "data/m_crop"
TRAIN_CSV = "data/m_crop/train.csv"
TEST_CSV = "data/m_crop/test.csv"
VALID_CSV = "data/m_crop/valid.csv"
modelPath = "model/" + sys.argv[1]
start = int(sys.argv[1].split(".")[0][5:])
EPOCHS = start + int(sys.argv[2])

train_dataset = DropletDataset(root_dir = ROOT_DIR, annotations_path = TRAIN_CSV)
test_dataset = DropletDataset(root_dir = ROOT_DIR, annotations_path = TEST_CSV)
valid_dataset = DropletDataset(root_dir = ROOT_DIR, annotations_path = VALID_CSV)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.load_state_dict(torch.load(modelPath))
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
criterion = torch.nn.MSELoss()



for epoch in range(start, EPOCHS+1):
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
            tepoch.set_postfix(mse=loss.item())
    
    with tqdm(valid_dataloader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Validation {epoch}")
            data, target = data.to(device), target.to(device)
           
            preds = model(data).squeeze()
            outputs = torch.sigmoid(preds)
            loss = criterion(target.float(), outputs)
            tepoch.set_postfix(mse = loss.item())
    if epoch%5 == 0:
        tqdm.write("Saving Model" + str(epoch) + ".torch")
        torch.save(model.state_dict(), 'model/' + 'model' + str(epoch) + ".torch")