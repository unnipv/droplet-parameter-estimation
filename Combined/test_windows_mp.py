from dataset.dropletsMP import DropletDataset
from torch.utils.data import DataLoader
from model.Unninet import UnniNet
from model.Soorkinet import SoorkiNet
import torch
import torchvision
from PIL import Image
import sys
import os

ROOT_DIR = r"data\crop"
TEST_CSV = r"data\crop\test.csv"
modelPath = os.path.join(r"model",  sys.argv[1])

#--------------Loading Test Data--------------------------------------------------------------------#
test_dataset = DropletDataset(root_dir = ROOT_DIR, annotations_path = TEST_CSV)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnniNet()
model = model.to(device)

#----------------Load State--------------------------------------------------------------------------
model.load_state_dict(torch.load(modelPath, map_location= device)) # Load trained model
model.eval() # Set net to evaluation mode, usually usefull in this case its fail

#----------------Make preidction--------------------------------------------------------------------------
for data, target in test_dataloader:
    data, target = data.to(device), target.to(device)           
    preds = model(data).squeeze()
    outputs = torch.sigmoid(preds)

criterion = torch.nn.MSELoss()
print("Predictions")
print(str(-1290 + outputs * 2800))

print("Labels")
print(str(-1290 + target * 2800))
print("MSE")
print(criterion(target.float(), outputs).item())
print(criterion((-1290 + outputs * 2800),(-1290 + target * 2800)))
#individual image
# default_transforms = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Grayscale(num_output_channels = 1),
#     torchvision.transforms.Normalize((0.5), (0.5)),
#     ]
# )
# img_path = sys.argv[2]
# image = Image.open(img_path)
# image = default_transforms(image).unsqueeze(0).to(device)
# print(image.size())
# pred = model(image)
# out = 20 + torch.sigmoid(pred) * 1380
# print(out)



