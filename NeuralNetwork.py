"""
Neural Network from scratch.
Class: Value, Neuron, Layer, MLP
"""

import math
import random


class Value:

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exp):
        out = Value(self.data**exp, (self,), f"**{exp}")

        def _backward():
            self.grad += exp * (self.data ** (exp - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other**-1)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = lambda self, o: Value(o) - self
    __rtruediv__ = lambda self, o: Value(o) / self

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class Neuron:
    """activation(w·x + b)."""

    ACTIVATIONS = {"tanh", "relu", "sigmoid", "linear"}

    def __init__(self, n_inputs, activation="tanh"):
        assert activation in self.ACTIVATIONS
        scale = math.sqrt(2.0 / n_inputs)
        self.w = [Value(random.gauss(0, scale)) for _ in range(n_inputs)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x):
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == "tanh":
            return z.tanh()
        elif self.activation == "relu":
            return z.relu()
        elif self.activation == "sigmoid":
            return z.sigmoid()
        else:
            return z  # linear

    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, n_inputs, n_neurons, activation="tanh"):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_neurons)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:

    def __init__(self, n_inputs, layer_sizes, activations=None):
        if activations is None:
            activations = ["tanh"] * (len(layer_sizes) - 1) + ["linear"]
        assert len(activations) == len(layer_sizes)

        dims = [n_inputs] + layer_sizes
        self.layers = [
            Layer(dims[i], dims[i + 1], activations[i]) for i in range(len(layer_sizes))
        ]

    def __call__(self, x):
        x = [xi if isinstance(xi, Value) else Value(xi) for xi in x]
        for layer in self.layers:
            x = layer(x)
            if not isinstance(x, list):
                x = [x]
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def step(self, lr=0.01):
        """Descente de gradient (SGD) : w ← w - lr * ∂L/∂w"""
        for p in self.parameters():
            p.data -= lr * p.grad

    @staticmethod
    def mse_loss(preds, targets):
        """Mean Squared Error : (1/N) Σ(ŷ - y)²"""
        n = len(preds)
        return sum(((p - t) ** 2 for p, t in zip(preds, targets)), Value(0.0)) * (1 / n)

    @staticmethod
    def binary_cross_entropy(preds, targets):
        """Binary Cross-Entropy"""
        eps = 1e-7
        losses = []
        for p, t in zip(preds, targets):
            pc = max(eps, min(1 - eps, p.data))
            losses.append(-(t * math.log(pc) + (1 - t) * math.log(1 - pc)))
        return Value(sum(losses) / len(losses))

    def train(self, X, y, epochs=100, lr=0.01, loss_fn="mse", verbose=10):
        """
        Train : forward → loss → zero_grad → backward → step.
        """
        history = []

        for epoch in range(1, epochs + 1):
            preds = [self(xi) for xi in X]
            if loss_fn == "mse":
                loss = self.mse_loss(preds, [Value(t) for t in y])
            elif loss_fn == "bce":
                loss = self.binary_cross_entropy(preds, y)
            else:
                raise ValueError(f"loss_fn inconnu: {loss_fn!r}")

            self.zero_grad()
            loss.backward()
            self.step(lr)
            history.append(loss.data)
            if verbose and epoch % verbose == 0:
                print(f"Epoch {epoch:4d}/{epochs}  loss = {loss.data:.6f}")

        return history


if __name__ == "__main__":

    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [0.0, 1.0, 1.0, 0.0]

    random.seed(42)
    model = MLP(2, [8, 8, 1], activations=["tanh", "tanh", "tanh"])
    print(model)

    history = model.train(X, y, epochs=200, lr=0.05, verbose=40)

    print("\nPrédictions finales :")
    for xi, yi in zip(X, y):
        p = model(xi)
        print(f"  {xi}  →  pred={p.data:.4f}  cible={yi}")
    print(f"\nLoss finale : {history[-1]:.6f}")
