from dataset.droplets import DropletDataset
from torch.utils.data import DataLoader
from model.Unninet import UnniNet
import torch
import sys

ROOT_DIR = "data/m_crop"
TEST_CSV = "data/m_crop/test.csv"
modelPath = "models//" + sys.argv[1]

#--------------Loading Test Data--------------------------------------------------------------------#
test_dataset = DropletDataset(root_dir = ROOT_DIR, annotations_path = TEST_CSV)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnniNet()
model = model.to(device)

#----------------Load State--------------------------------------------------------------------------
model.load_state_dict(torch.load(modelPath)) # Load trained model
model.eval() # Set net to evaluation mode, usually usefull in this case its fail

#----------------Make preidction--------------------------------------------------------------------------
for data, target in test_dataloader:
    data, target = data.to(device), target.to(device)           
    preds = model(data)
    outputs = torch.sigmoid(preds)
    outputs = outputs * 1400

print("Predictions //n")
print(outputs)

