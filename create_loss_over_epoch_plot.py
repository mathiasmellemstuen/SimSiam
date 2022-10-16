import matplotlib.pyplot as plt
import numpy as np

def create_loss_over_epoch_plot(loss_with_stop_grad_file_name, loss_without_stop_grad_file_name):

    loss_with_stop_grad = read_loss_file(loss_with_stop_grad_file_name)
    loss_without_stop_grad = read_loss_file(loss_without_stop_grad_file_name)

    plt.figure()
    plt.ylabel("Training loss")
    plt.xlabel("Epochs")

    plt.axis([1, len(loss_with_stop_grad) + 1, -1.0, -0.5])

    x = np.arange(1, len(loss_with_stop_grad) + 1)
    plt.plot(x, loss_with_stop_grad, color="blue", label="w/ stop-gradient")

    x = np.arange(1, len(loss_without_stop_grad) + 1)
    plt.plot(x, loss_without_stop_grad, color="red", label="w/o stop-gradient")
    plt.legend()

def read_loss_file(file_path):
    file = open(file_path)
    lines = file.readlines()
    file.close()

    losses = []

    for line in lines:
        losses.append(float(line))

    return losses

if __name__ == "__main__":
    create_loss_over_epoch_plot("loss_with_stop_gradient.txt", "loss_without_stop_gradient.txt")
    plt.show()