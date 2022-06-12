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

ROOT_DIR = r"200"
TEST_CSV = r"200\labels.csv"
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

#----------------Make preidction--------------------------------------------------------------------------
dia = []
for data, target in test_dataloader:
    data, target = data.to(device), target.to(device)           
    preds = model(data).squeeze()
    outputs = torch.sigmoid(preds)
    target = target.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    dia_pred = 50 + outputs * 350
    dia_label = 50 + target[0] * 350
    dia.append([dia_label, dia_pred])
print(np.array(dia).shape)
dia_pd = pd.DataFrame(dia, columns = ["Label", "Predicted"])
dia_pd.to_csv(r"model\train_no_200_1\dia200.csv")



