import numpy as np

from .neural_networks import NeuralNetwork
from .dataloaders import Dataloader


def test(model: NeuralNetwork, data: Dataloader):
    """
    Steps:
        * for each input data,
            * pass forward on the model
            * calculate accuracy
        * output the average accuracy of the model
    """
    correct = 0
    for idx in range(len(data)):
        sample = data.get_sample(idx)

        pred_y = model.forward(sample["input"])
        pred_class = np.argmax(pred_y)

        true_y = sample["label"]
        true_class = np.argmax(true_y)

        if true_class == pred_class:
            correct += 1

    print("The Accuracy of the model is: ", correct / len(data) * 100)
