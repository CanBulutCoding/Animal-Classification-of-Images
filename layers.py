import pickle

import numpy as np


class Layer:

    def forward(self, x):
        pass

    def backward(self, dout):
        pass

    def update_weights(self, lr: float):
        pass

    def zero_grad(self):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass


class FullyConnectedLayer(Layer):

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()

        self.weights = np.random.random(size=(input_size, output_size)) - 0.5
        self.bias = np.random.random(size=(1, output_size)) - 0.5

        self.gradients = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        out = np.matmul(x, self.weights) + self.bias
        return out

    def backward(self, dout: np.ndarray):
        dw = self.x.T.dot(dout)
        db = dout

        self.gradients.append((dw, db))

        dx = dout.dot(self.weights.T)

        return dx

    def update_weights(self, lr: float):
        dw = 0
        db = 0
        for w, b in self.gradients:
            dw += w
            db += b
        dw = dw / len(self.gradients)
        db = db / len(self.gradients)

        self.weights = self.weights - (lr * dw)
        self.bias = self.bias - (lr - db)

    def zero_grad(self):
        self.gradients = []

    def save(self, path: str) -> None:
        params = {"w": self.weights, "b": self.bias}
        s_params = pickle.dumps(params)

        with open(path, "wb") as f:
            f.write(s_params)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            s_params = f.read()
        params = pickle.loads(s_params)

        self.weights = params["w"]
        self.bias = params["b"]


class SigmoidLayer(Layer):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y

    def backward(self, dout: float):
        ds = self.y * (1 - self.y)
        return ds * dout
