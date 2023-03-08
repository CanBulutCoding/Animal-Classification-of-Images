from typing import List

import numpy as np

from .layers import Layer


class NeuralNetwork:

    def forward(self):
        pass

    def backward(self):
        pass


class MultiLayerNeuralNetwork(NeuralNetwork):

    def __init__(self) -> None:
        self.layers = []

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dl: float) -> None:
        out = dl
        for layer in reversed(self.layers):
            out = layer.backward(out)

    def update_weights(self, lr: float) -> None:
        for layer in self.layers:
            layer.update_weights(lr=lr)

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def save(self, dir_path: str) -> None:
        for idx, layer in enumerate(self.layers):
            weight_path = f"{dir_path}/layer_{idx}.weights"
            layer.save(weight_path)

    def load(self, dir_path: str) -> None:
        for idx, layer in enumerate(self.layers):
            weight_path = f"{dir_path}/layer_{idx}.weights"
            layer.load(weight_path)
