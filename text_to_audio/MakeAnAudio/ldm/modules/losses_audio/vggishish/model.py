import torch
import torch.nn as nn


class VGGishish(nn.Module):

    def __init__(self, conv_layers, use_bn, num_classes):
        '''
        Mostly from
            https://pytorch.org/vision/0.8/_modules/torchvision/models/vgg.html
        '''
        super().__init__()
        layers = []
        in_channels = 1

        # a list of channels with 'MP' (maxpool) from config
        for v in conv_layers:
            if v == 'MP':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=1)
                if use_bn:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.features = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((5, 10))

        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 5 * 10, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )

        # weight init
        self.reset_parameters()

    def forward(self, x):
        # adding channel dim for conv2d (B, 1, F, T) <-
        x = x.unsqueeze(1)
        # backbone (B, 1, 5, 53) <- (B, 1, 80, 860)
        x = self.features(x)
        # adaptive avg pooling (B, 1, 5, 10) <- (B, 1, 5, 53) – if no MP is used as the end of VGG
        x = self.avgpool(x)
        # flatten
        x = self.flatten(x)
        # classify
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    num_classes = 309
    inputs = torch.rand(3, 80, 848)
    conv_layers = [64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 'MP', 512, 512, 512, 'MP', 512, 512, 512]
    # conv_layers = [64, 'MP', 128, 'MP', 256, 256, 'MP', 512, 512, 'MP']
    model = VGGishish(conv_layers, use_bn=False, num_classes=num_classes)
    outputs = model(inputs)
    print(outputs.shape)
