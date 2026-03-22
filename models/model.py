import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Take as input a 3 Channels x 64 W x 64 H

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        # Going from 3 to 64 channels,
        # kernel size=3 and padding=1
        # allows the layer to extract features without changing the size of the image.

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # At this point I got a 128x64x64

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # At this point I got a 256x64x64

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # At this point I got a 512x64x64

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # This transforms 512x64x64 -> 512x1x1

        self.fc1 = nn.Linear(512, 200)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.maxpool(torch.relu(self.conv1(x))) # 64x64 -> 32x32 (if using 64x64 input), thanks to maxpool
        x = self.maxpool(torch.relu(self.conv2(x))) # 32x32 -> 16x16
        x = self.maxpool(torch.relu(self.conv3(x))) # 16x16 -> 8x8
        x = self.maxpool(torch.relu(self.conv4(x))) # 8x8 -> 4x4

        x = self.pool(x)        # Becomes [Batch, 256, 1, 1]

        x = torch.flatten(x, 1) # Becomes [Batch, 256]

        return self.fc1(x)      # Final 200 class scores