import math
import random

class Value:

    
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda:None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += self.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, exp):
        out = Value(self.data ** exp, (self,), f"**{exp}")
        def _backward():
            self.grad += exp * (self.data ** (exp - 1)) * out.grad
        out._backward = _backward
        return out
    
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

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    

class Neuron:

    ACTIVATIONS = {"tanh", "relu", "sigmoid", "linear"}

    def __init__(self, n_inputs, activation="tanh"):
        assert activation in self.ACTIVATIONS
        scale = math.sqrt(2.0 / n_inputs)
        self.w = [Value(random.gauss(0, scale)) for _ in range(n_inputs)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x):
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if   self.activation == "tanh":    return z.tanh()
        elif self.activation == "relu":    return z.relu()
        elif self.activation == "sigmoid": return z.sigmoid()
        else:                              return z          # linear

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron(in={len(self.w)}, act={self.activation})"
