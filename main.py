"""
Steps:
    * construct dataloader for train and test
    * construct model
    * run the training loop
    * save the model
    * test the mdoel
"""
import warnings

import numpy as np

from code.dataloaders import AnimalsDataloader
from code.neural_networks import MultiLayerNeuralNetwork
from code.layers import FullyConnectedLayer, SigmoidLayer
from code.training import train
from code.testing import test
from code.losses import nll_loss, nll_loss_derivative


warnings.filterwarnings("ignore")


animals_train_data = AnimalsDataloader(data_dir="data/train", num_class=10)
animals_test_data = AnimalsDataloader(data_dir="data/test", num_class=10)

# for i in range(5):
#     sample = animals_test_data.get_sample(i)
#     print(sample["input"].shape)
#     print(sample["label"])

animals_model = MultiLayerNeuralNetwork()
animals_model.add_layer(FullyConnectedLayer(input_size=65536, output_size=10))
animals_model.add_layer(SigmoidLayer())

# animals_model.load("models/latest")

for i in range(5):
    train(model=animals_model, data=animals_train_data, batch_size=4, loss_fn=nll_loss, d_loss_fn=nll_loss_derivative, num_epochs=2, lr=0.005)

    animals_model.save("models/latest")

    test(animals_model, data=animals_test_data)
