from dataset.dropletsMP import DropletDataset
from torch.utils.data import DataLoader
from model.Unninet import UnniNet
import torch
import torchvision
from PIL import Image
import sys
import os
import numpy as np
import pandas as pd

ROOT_DIR = r"data\crop"
TEST_CSV = r"data\crop\test.csv"
modelPath = os.path.join(r"model",  sys.argv[1])

#--------------Loading Test Data--------------------------------------------------------------------#
test_dataset = DropletDataset(root_dir = ROOT_DIR, annotations_path = TEST_CSV)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnniNet()
model = model.to(device)

#----------------Load State--------------------------------------------------------------------------
model.load_state_dict(torch.load(modelPath, map_location= device)) # Load trained model
model.eval() # Set net to evaluation mode, usually usefull in this case its fail

#----------------Make preidction--------------------------------------------------------------------------

pos = []
for data, target in test_dataloader:
    data, target = data.to(device), target.to(device)           
    preds = model(data).squeeze()
    outputs = torch.sigmoid(preds)
    target = target.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    pos_pred = -1290 + outputs * 2800
    pos_label = -1290 + target[0] * 2800
    pos.append([pos_label, pos_pred])
print(np.array(pos).shape)
pos_pd = pd.DataFrame(pos, columns = ["Label", "Predicted"])
pos_pd.to_csv(r"model\train_train_mp_29May\pos.csv")


