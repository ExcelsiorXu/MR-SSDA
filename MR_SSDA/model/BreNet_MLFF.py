import torch
import torch.nn as nn
import torch.nn.functional as F
import MyModel
# complete algorithm

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        model = MyModel.BreNet().cuda()
        model.load_state_dict(torch.load('./model/BreNet_PreTrain.pkl'))
        self.features = model.features
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x):
        x1 = x
        x2 = x
        x3 = x
        for i in range(25):
            x1 = self.features[i](x1)
        for i in range(20):
            x2 = self.features[i](x2)
        for i in range(15):
            x3 = self.features[i](x3)
        x1 = F.avg_pool2d(x1, kernel_size=x1.size()[2:])
        x2 = F.avg_pool2d(x2, kernel_size=x2.size()[2:])
        x3 = F.avg_pool2d(x3, kernel_size=x3.size()[2:])
        x123 = torch.cat([torch.cat([x1, x2], dim=1), x3], dim=1)
        out = x123
        return out

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1152, 512),
            nn.BatchNorm1d(512))
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128))
        self.fc3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 2),
            nn.BatchNorm1d(2))
        self.sig = nn.Sigmoid()
        self.bn_fc = nn.BatchNorm1d(1152)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.MLFF = MyModel.MLFFLayer(1152, 512)

    def forward(self, x):
        MLFF_in = x
        x = x.view(x.size(0), -1)
        s_bn1 = self.bn_fc(x)
        MLFF_out = self.MLFF(MLFF_in)
        x_out = MLFF_out.view(MLFF_out.size(0), -1)
        s_bn2 = self.bn_fc2(x_out)
        x = F.relu(x_out)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x_sig = self.sig(x)
        return x_sig, s_bn1, MLFF_in, MLFF_out, s_bn2








