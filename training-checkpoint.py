from typing import Callable

from .neural_networks import NeuralNetwork
from .dataloaders import Dataloader


def train(model: NeuralNetwork, data: Dataloader, batch_size: int, loss_fn: Callable, d_loss_fn: Callable, num_epochs: int, lr: float):
    """
    Steps:
        * for every epoch,
            * for every input in the dataset,
                * get output from the model by passing it forward
                * calculate loss
                * backpropagate the loss to calculate the gradients
                * if batch size is reached,
                    * update the weights
                    * zero the gradients
    """
    for epoch in range(num_epochs):
        losses = []
        for idx in range(len(data)):
            sample = data.get_sample(index=idx)

            pred_y = model.forward(sample["input"])

            loss = loss_fn(sample["label"], pred_y)
            dl = d_loss_fn(sample["label"], pred_y)

            model.backward(dl)
            losses.append(loss)


            if (idx + 1) % batch_size == 0:
                model.update_weights(lr=lr)
                model.zero_grad()

        print("The average loss for the epoch is: ", sum(losses) / len(losses))
