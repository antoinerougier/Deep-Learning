"""
Tests unitaires pour la classe Value.
"""
import math
import pytest
from NeuralNetwork import Value


# ── Opérations de base ────────────────────────────────────────────────────────

def test_add():
    a = Value(2.0)
    b = Value(3.0)
    assert (a + b).data == 5.0

def test_mul():
    a = Value(3.0)
    b = Value(4.0)
    assert (a * b).data == 12.0

def test_sub():
    a = Value(5.0)
    b = Value(2.0)
    assert (a - b).data == 3.0

def test_div():
    a = Value(6.0)
    b = Value(2.0)
    assert (a / b).data == pytest.approx(3.0)

def test_pow():
    a = Value(3.0)
    assert (a ** 2).data == pytest.approx(9.0)

def test_neg():
    a = Value(4.0)
    assert (-a).data == -4.0


# ── Opérateurs réfléchis (radd, rmul…) ───────────────────────────────────────

def test_radd():
    a = Value(3.0)
    assert (2.0 + a).data == 5.0

def test_rmul():
    a = Value(3.0)
    assert (4.0 * a).data == 12.0


# ── Fonctions d'activation ────────────────────────────────────────────────────

def test_tanh():
    a = Value(0.0)
    assert a.tanh().data == pytest.approx(0.0)

    a = Value(1.0)
    assert a.tanh().data == pytest.approx(math.tanh(1.0))

def test_relu_positive():
    a = Value(3.0)
    assert a.relu().data == 3.0

def test_relu_negative():
    a = Value(-2.0)
    assert a.relu().data == 0.0

def test_sigmoid_zero():
    a = Value(0.0)
    assert a.sigmoid().data == pytest.approx(0.5)

def test_sigmoid_range():
    for x in [-10.0, -1.0, 0.0, 1.0, 10.0]:
        s = Value(x).sigmoid().data
        assert 0.0 < s < 1.0


# ── Backward (gradients) ──────────────────────────────────────────────────────

def test_backward_add():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()
    assert a.grad == pytest.approx(1.0)
    assert b.grad == pytest.approx(1.0)

def test_backward_mul():
    a = Value(3.0)
    b = Value(4.0)
    c = a * b
    c.backward()
    assert a.grad == pytest.approx(4.0)  # ∂(a*b)/∂a = b
    assert b.grad == pytest.approx(3.0)  # ∂(a*b)/∂b = a

def test_backward_pow():
    a = Value(3.0)
    c = a ** 2
    c.backward()
    assert a.grad == pytest.approx(6.0)  # ∂(a²)/∂a = 2a = 6

def test_backward_tanh():
    a = Value(0.0)
    a.tanh().backward()
    assert a.grad == pytest.approx(1.0)  # 1 - tanh(0)² = 1

def test_backward_relu_positive():
    a = Value(2.0)
    a.relu().backward()
    assert a.grad == pytest.approx(1.0)

def test_backward_relu_negative():
    a = Value(-1.0)
    a.relu().backward()
    assert a.grad == pytest.approx(0.0)

def test_backward_chain():
    # f = (a + b) * c  →  ∂f/∂a = c, ∂f/∂b = c, ∂f/∂c = a+b
    a = Value(2.0)
    b = Value(3.0)
    c = Value(4.0)
    f = (a + b) * c
    f.backward()
    assert a.grad == pytest.approx(4.0)
    assert b.grad == pytest.approx(4.0)
    assert c.grad == pytest.approx(5.0)