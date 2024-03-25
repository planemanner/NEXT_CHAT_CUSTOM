from torch import nn
from transformers import SwinModel, AutoImageProcessor


def disable_learning(model : nn.Module):
    # To freeze a pretrained model
    for param in model.parameters():
        param.requires_grad = False


# from_what = 'microsoft/swin-tiny-patch4-window7-224'
# image_processor = AutoImageProcessor.from_pretrained(from_what)
# model = SwinModel.from_pretrained(from_what)

# disable_learning(model)