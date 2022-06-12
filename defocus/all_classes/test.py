from dataset.droplets import DropletDataset
from torch.utils.data import DataLoader
from model.dropletnet import DropletNet
import torch
import torchvision
from PIL import Image
import sys
import os
import numpy as np
import pandas as pd

ROOT_DIR = r"data"
TEST_CSV = r"data\test.csv"
modelPath = os.path.join(r"model",  sys.argv[1])

#--------------Loading Test Data--------------------------------------------------------------------#
test_dataset = DropletDataset(root_dir = ROOT_DIR, annotations_path = TEST_CSV)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DropletNet()
model = model.to(device)

#----------------Load State--------------------------------------------------------------------------
model.load_state_dict(torch.load(modelPath, map_location= device)) # Load trained model
model.eval() # Set net to evaluation mode, usually usefull in this case its fail

#----------------Make prediction--------------------------------------------------------------------------
dia = []
pos = []
for data, target in test_dataloader:
    data, target = data.to(device), target.to(device)           
    preds = model(data).squeeze()
    outputs = torch.sigmoid(preds)
    target = target.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    dia_pred = 50 + outputs[0] * 350
    pos_pred = -300 + outputs[1] * 600
    dia_label = 50 + target[0,0] * 350
    pos_label = -300 + target[0,1] * 600
    dia.append([dia_label, dia_pred])
    pos.append([pos_label, pos_pred])
print(np.array(dia).shape)
print(np.array(pos).shape)
dia_pd = pd.DataFrame(dia, columns = ["Label", "Predicted"])
pos_pd = pd.DataFrame(pos, columns = ["Label", "Predicted"])
dia_pd.to_csv(r"model\train_all_classes_1\dia.csv")
pos_pd.to_csv(r"model\train_all_classes_1\pos.csv")



