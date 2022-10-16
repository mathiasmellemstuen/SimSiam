import torch
import torch.nn as nn
import torchvision.models as models

class SimSiamNetwork(nn.Module):
    def __init__(self, model_architecture_name, dimention, prediction_dimention, stop_gradient = True) -> None:
        super().__init__()

        self.stop_gradient = stop_gradient
        self.encoder = models.__dict__[model_architecture_name](num_classes=dimention, zero_init_residual=True)

        size = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(size, size, bias=False), nn.BatchNorm1d(size), nn.ReLU(inplace=True), nn.Linear(size, size, bias=False), nn.BatchNorm1d(size), nn.ReLU(inplace=True), self.encoder.fc, nn.BatchNorm1d(dimention, affine=False))
        self.encoder.fc[6].bias.requires_grad = False

        self.predictor = nn.Sequential(nn.Linear(dimention, prediction_dimention, bias = False), nn.BatchNorm1d(prediction_dimention), nn.ReLU(inplace=True), nn.Linear(prediction_dimention, dimention)) 
    
    def forward(self, image_view_1, image_view_2):

        features_image_view_1 = self.encoder(image_view_1)
        features_image_view_2 = self.encoder(image_view_2)

        prediction_image_view_1 = self.predictor(features_image_view_1)
        prediction_image_view_2 = self.predictor(features_image_view_2)


        # The detach function is working as the stop-grad operation in the SimSiam algorithm.
        if self.stop_gradient:
            return prediction_image_view_1, prediction_image_view_2, features_image_view_1.detach(), features_image_view_2.detach()
        else:
            return prediction_image_view_1, prediction_image_view_2, features_image_view_1, features_image_view_2