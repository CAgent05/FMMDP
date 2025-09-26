from Baselines.ResNet import *
import torch
from Prototypical_Loss import prototypical_testing as ptest

model_resnet = ResNet(input_size = 20, nb_classes=2)
model_resnet.load_state_dict(torch.load('/home/jc/CY/FMMDP/model/COSCO/BipedalWalkerHCSA_20.pth'))
model_resnet.to('cuda:0')
model_resnet.eval()

train_data = torch.load('/home/jc/CY/FMMDP/data/Train/BipedalWalkerHCSA_20/X_train.pt').squeeze(1)
train_label = torch.load('/home/jc/CY/FMMDP/data/Train/BipedalWalkerHCSA_20/y_train.pt')
test_data = torch.load('/home/jc/CY/FMMDP/data/Train/BipedalWalkerHCSA_20/X_valid.pt').squeeze(1)
test_label = torch.load('/home/jc/CY/FMMDP/data/Train/BipedalWalkerHCSA_20/y_valid.pt')


pre, embd = model_resnet(test_data.transpose(1,2).float())

train_centroids = torch.load('train_centroids.pt')

predicted_test_labels = ptest(embd, train_centroids)

correct = (predicted_test_labels.cuda() == test_label.cuda()).sum().item()
print(correct/test_label.size(0))