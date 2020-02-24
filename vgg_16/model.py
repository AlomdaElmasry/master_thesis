import ssl
import torch
import torch.nn as nn
import torchvision.models
from torchvision.models.vgg import cfgs, make_layers, model_urls

ssl._create_default_https_context = ssl._create_unverified_context


class VGGFeatures(torchvision.models.VGG):

    def forward(self, x):
        pool_feats = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                pool_feats.append(x.detach())
        # x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, pool_feats


def get_pretrained_model():
    model = VGGFeatures(make_layers(cfgs['D'], batch_norm=False))
    state_dict = torchvision.models.utils.load_state_dict_from_url(model_urls['vgg16'])
    model.load_state_dict(state_dict)
    return model
