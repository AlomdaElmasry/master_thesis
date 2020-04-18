from models.vgg_16 import get_pretrained_model
import torch


class PerceptualUtils:
    model_vgg = None

    def __init__(self, device):
        self.model_vgg = get_pretrained_model(device)
        for param in self.model_vgg.features.parameters():
            param.requires_grad = False

    def vgg_features(self, y):
        return self.model_vgg(y.contiguous())
