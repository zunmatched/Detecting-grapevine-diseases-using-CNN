import torchvision
import torch
import torch.nn as nn
# from PIL import Image

def accuracy(out, yb):
    '''計算準確率'''
    return torch.mean((torch.argmax(out, dim=1) == torch.argmax(yb, dim=1)).float())

class Layer_SPP(nn.Module):
    '''SPP: Spatial Pyramid Pooling'''
    def __init__(self, bin = [1, 2, 4]):
        super().__init__()
        self.model = nn.Module() ## 儲存pool層的容器
        for i, b in enumerate(bin):
            self.model.add_module('aap2d_{}'.format(i), nn.AdaptiveAvgPool2d(b))
        pass
    def forward(self, x):
        output = torch.cat([torch.flatten(b(x), -3, -1) for b in self.model.children()], -1)
        return output
        pass

class gvd_detect(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
            nn.ReLU(), 
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
            nn.ReLU(), 
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )
        self.spp = Layer_SPP([2, 8])
        self.classifier = nn.Sequential(
            nn.LayerNorm(2176), nn.Dropout(), 
            nn.Linear(2176, 1024), 
            nn.LeakyReLU(), nn.Dropout(), 
            nn.Linear(1024 , 256), 
            nn.LeakyReLU(), 
            nn.Linear(256, 64), 
            nn.LeakyReLU(), 
            nn.Linear(64, 4), 
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.spp(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = gvd_detect()
    m = m.to(device)
    print(m)
    input = torch.rand(5, 3, 128, 128)
    input = input.to(device)
    output = m(input)
    print(output.size())
    print(output)
    pass