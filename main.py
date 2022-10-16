from SimSiamNetwork import SimSiamNetwork
import torch.nn as nn
import torch.cuda as cuda
import torchvision.datasets as datasets
import torch.utils.data
from split_in_two import SplitInTwo
import torchvision.transforms as transforms
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from create_loss_over_epoch_plot import create_loss_over_epoch_plot

def write_progress_to_file(losses, file_name): 
    with open(file_name, "w") as file:
        for line in losses:
            file.write(f"{line}\n")

def setup_and_train(device, dimention, prediction_dimention, max_epochs, train_directory, stop_gradient, loss_file_name):

    # Setting up the network with resnet50
    network = SimSiamNetwork(model_architecture_name = "resnet50", dimention=dimention, prediction_dimention=prediction_dimention, stop_gradient=stop_gradient)
    network.to(device)


    # All augmentations for each image
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(int(64 * 0.1) + (int(64 * 0.1) - 1) % 2)
        ], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Applying augmentations and splitting loaded images into two augmented views
    training_data = datasets.ImageFolder(train_directory, SplitInTwo(transform))

    training_loader = torch.utils.data.DataLoader(training_data, batch_size=16, shuffle=True, num_workers=8)

    network.train()

    criterion = nn.CosineSimilarity().cuda(device)
    
    optimizer = Adam(params=network.parameters(), lr=4e-4)

    epoch_losses = []

    for epoch in range(max_epochs):
        losses = []
        for step, (images, y) in enumerate(training_loader):
            
            # Two augmented views of the same image
            image_1, image_2 = images[0], images[1]

            image_1 = image_1.cuda(device)
            image_2 = image_2.cuda(device)


            # Forward pass in the SimSiam network with the two augmented views (forward propagation)
            prediction_image_view_1, prediction_image_view_2, feature_image_view_1, feature_image_view_2 = network(image_1, image_2)

            # Calculating loss from the output of the forward pass
            loss = -(criterion(prediction_image_view_1, feature_image_view_2).mean() + criterion(prediction_image_view_1, feature_image_view_2).mean()) * 0.5

            losses.append(loss.item())

            # Backward pass in the neural network (backpropagation)
            loss.backward()

            optimizer.step()

            print(f"Epoch {epoch} progress: {round((step / len(training_loader)) * 100, 2)}% | {loss.item()}")

        epoch_losses.append(np.mean(losses))
        losses = []

        # Writing losses to file
        write_progress_to_file(epoch_losses, loss_file_name)

if __name__ == "__main__": 

    torch.cuda.empty_cache()

    dimention = 2048
    prediction_dimention = 512
    max_epochs = 50
    train_directory = "tiny-imagenet-200/train"
    device = "cuda" if cuda.is_available() else "cpu"

    if not device == "cuda":
        print("Cuda not supported. Using CPU")

    # Running SimSiam network with the stop-gradient operation
    setup_and_train(device, dimention, prediction_dimention, max_epochs, train_directory, stop_gradient=True, loss_file_name="loss_with_stop_gradient.txt")

    # Running SimSiam network without the stop-gradient operation (to explore the effects of the stop-gradient operation)
    setup_and_train(device, dimention, prediction_dimention, max_epochs, train_directory, stop_gradient=False, loss_file_name="loss_without_stop_gradient.txt")

    # Plotting both results to the same plot
    create_loss_over_epoch_plot("loss_with_stop_gradient.txt", "loss_without_stop_gradient.txt")
    plt.show()