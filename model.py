import torch
import torch.nn as nn

class SiameseNN(nn.Module):

    def __init__(self):
        super(SiameseNN, self).__init__()
        self.conv_layer = nn.Sequential(
                            nn.Conv2d(1, 64, 10),
                            nn.ReLU(),
                            nn.MaxPool2d(2, 2),
                            nn.Conv2d(64, 128, 7),
                            nn.ReLU(),
                            nn.MaxPool2d(2, 2),
                            nn.Conv2d(128, 128, 4),
                            nn.ReLU(),
                            nn.MaxPool2d(2, 2),
                            nn.Conv2d(128, 256, 4),
                            nn.ReLU()
                            )

        self.fc_layer = nn.Sequential(
                        nn.Linear(256 * 6 * 6, 4096), 
                        nn.Sigmoid())
        self.fc_out    = nn.Sequential(
                        nn.Linear(4096, 1) ,
                        nn.Sigmoid())

    def encoder(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(torch.flatten(x, start_dim=1))
        return x

    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.normal_(m.bias, mean=0.5, std=0.01)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.2)
            nn.init.normal_(m.bias, mean=0.5, std=0.01)

    
    def forward(self, img1, img2):
        hidden1 = self.encoder(img1)
        hidden2 = self.encoder(img2)
        dist    = torch.abs(hidden1 - hidden2)
        out     = self.fc_out(dist)
        return out

if __name__ == '__main__':
    model = SiameseNN()
    print(model)
    print(model.state_dict().keys())