import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG19_Weights

class ResidualBlock(nn.Module):
    """
    ResidualBlock Class
    Values
        channels: the number of channels throughout the residual block, a scalar
    """

    def __init__(self, channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=False),

            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=False),
        )

    def forward(self, x):
        return x + self.layers(x)


class Encoder(nn.Module):
    '''
    Encoder Class
    Values:
        in_channels: number of input channels to each discriminator, a scalar
        out_channels: number of channels in output feature map, a scalar
        base_channels: number of channels in first convolutional layer, a scalar
        n_layers: number of downsampling layers, a scalar
    '''

    def __init__(self, in_channels, out_channels, base_channels=16, n_layers=4):
        super().__init__()

        self.out_channels = out_channels
        channels = base_channels

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=0), 
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        ]

        # Downsampling layers
        for i in range(n_layers):
            layers += [
                nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(2 * channels),
                nn.ReLU(inplace=True),
            ]
            channels *= 2
    
        # Upsampling layers
        for i in range(n_layers):
            layers += [
                nn.ConvTranspose2d(channels, channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(channels // 2),
                nn.ReLU(inplace=True),
            ]
            channels //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.layers = nn.Sequential(*layers)

    def instancewise_average_pooling(self, x, inst):
        '''
        Applies instance-wise average pooling.

        Given a feature map of size (b, c, h, w), the mean is computed for each b, c
        across all h, w of the same instance
        '''
        x_mean = torch.zeros_like(x)
        classes = torch.unique(inst, return_inverse=False, return_counts=False) # gather all unique classes present

        for i in classes:
            for b in range(x.size(0)):
                indices = torch.nonzero(inst[b:b+1] == i, as_tuple=False) # get indices of all positions equal to class i
                for j in range(self.out_channels):
                    x_ins = x[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]]
                    mean_feat = torch.mean(x_ins).expand_as(x_ins)
                    x_mean[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]] = mean_feat

        return x_mean

    def forward(self, x, inst):
        x = self.layers(x)
        x = self.instancewise_average_pooling(x, inst)
        return x


class VGG19(nn.Module):
    '''
    VGG19 Class
    Wrapper for pretrained torchvision.models.vgg19 to output intermediate feature maps
    '''

    def __init__(self):
        super().__init__()
        #vgg_features = models.vgg19(pretrained=False).features
        vgg_features = models.vgg19(weights=VGG19_Weights.DEFAULT).features # line above will be deprecated after torchvision 0.15
        
        self.f1 = nn.Sequential(*[vgg_features[x] for x in range(2)])
        self.f2 = nn.Sequential(*[vgg_features[x] for x in range(2, 7)])
        self.f3 = nn.Sequential(*[vgg_features[x] for x in range(7, 12)])
        self.f4 = nn.Sequential(*[vgg_features[x] for x in range(12, 21)])
        self.f5 = nn.Sequential(*[vgg_features[x] for x in range(21, 30)])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h1 = self.f1(x)
        h2 = self.f2(h1)
        h3 = self.f3(h2)
        h4 = self.f4(h3)
        h5 = self.f5(h4)
        return [h1, h2, h3, h4, h5]

        