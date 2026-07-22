"""Focused correctness tests for the activation primitives from q01."""

import numpy as np
import pytest

from bert_cpu import engine as cpu
from exercises.q01_activations import ExTensor
from test.test_engine import numeric_gradient


ACTIVATIONS = ("sigmoid", "swish", "softplus")


def _sigmoid_reference(values: np.ndarray) -> np.ndarray:
    """Independent, stable NumPy reference used by the tests."""
    result = np.empty_like(values)
    non_negative = values >= 0.0
    result[non_negative] = 1.0 / (1.0 + np.exp(-values[non_negative]))
    exp_x = np.exp(values[~non_negative])
    result[~non_negative] = exp_x / (1.0 + exp_x)
    return result


def _reference(name: str, values: np.ndarray) -> np.ndarray:
    sigmoid = _sigmoid_reference(values)
    if name == "sigmoid":
        return sigmoid
    if name == "swish":
        return values * sigmoid
    if name == "softplus":
        return np.logaddexp(0.0, values)
    raise AssertionError(f"unknown activation: {name}")


def _local_derivative(name: str, values: np.ndarray) -> np.ndarray:
    sigmoid = _sigmoid_reference(values)
    if name == "sigmoid":
        return sigmoid * (1.0 - sigmoid)
    if name == "swish":
        return sigmoid + values * sigmoid * (1.0 - sigmoid)
    if name == "softplus":
        return sigmoid
    raise AssertionError(f"unknown activation: {name}")


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_values_shape_dtype_and_graph_metadata(name):
    values = np.array([[-3.0, -0.5, 0.0], [0.5, 2.0, 5.0]], dtype=np.float32)
    x = ExTensor(values)

    out = getattr(x, name)()

    assert np.allclose(out.data, _reference(name, values), rtol=1e-6, atol=1e-7)
    assert out.shape == x.shape
    assert out.data.dtype == values.dtype
    assert out.requires_grad is True
    assert out._op == name
    assert out._prev == {x}


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_extreme_inputs_and_derivatives_are_finite(name):
    values = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0])
    x = ExTensor(values)

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        out = getattr(x, name)()
        out.sum().backward()

    assert np.all(np.isfinite(out.data))
    assert np.all(np.isfinite(x.grad))
    assert np.allclose(out.data, _reference(name, values))
    assert np.allclose(x.grad, _local_derivative(name, values))


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_gradient_matches_central_differences(name):
    values = np.array([-2.3, -0.7, 0.2, 1.8])
    weights = np.array([0.5, -1.2, 2.0, 0.7])
    x = ExTensor(values)

    (getattr(x, name)() * weights).sum().backward()
    numerical = numeric_gradient(
        lambda: float((_reference(name, x.data) * weights).sum()),
        x.data,
    )

    assert np.allclose(x.grad, numerical, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_unbound_method_keeps_a_base_tensor_connected(name):
    """A Tensor made by base ops must stay connected through ExTensor.method."""
    values = np.array([-1.5, -0.1, 0.6, 2.0])
    weights = np.array([1.0, -0.25, 0.5, 1.5])
    x = cpu.Tensor(values)
    z = x * 1.7 + 0.2

    activated = getattr(ExTensor, name)(z)
    (activated * weights).sum().backward()
    numerical = numeric_gradient(
        lambda: float((_reference(name, 1.7 * x.data + 0.2) * weights).sum()),
        x.data,
    )

    assert type(z) is cpu.Tensor
    assert z in activated._prev
    assert np.allclose(x.grad, numerical, rtol=1e-5, atol=1e-6)


def test_given_softplus_neuron_equation_gradient():
    """The q01 g3 vector-dot chain works with the unbound Softplus method."""
    x = ExTensor(np.array([-0.8, 0.3, 1.2]))
    w = ExTensor(np.array([0.5, -1.0, 0.7]))
    b = ExTensor(np.array(-0.2))

    def forward():
        return ExTensor.softplus(x @ w + b).sum()

    forward().backward()

    for tensor in (x, w, b):
        numerical = numeric_gradient(lambda: float(forward().data), tensor.data)
        assert np.allclose(tensor.grad, numerical, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_gradient_accumulates_from_two_graph_paths(name):
    values = np.array([-1.0, 0.0, 2.0])
    x = ExTensor(values)
    activated = getattr(x, name)()

    (activated + activated).sum().backward()

    expected = 2.0 * _local_derivative(name, values)
    assert np.allclose(x.grad, expected)


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_requires_grad_false_is_preserved(name):
    x = ExTensor(np.array([-1.0, 0.0, 1.0]), requires_grad=False)

    out = getattr(x, name)()
    out.sum().backward()

    assert out.requires_grad is False
    assert np.array_equal(x.grad, np.zeros_like(x.data))


@pytest.mark.parametrize(
    ("name", "flops_per_element"),
    (("sigmoid", 4), ("swish", 5), ("softplus", 3)),
)
def test_forward_flop_convention(name, flops_per_element):
    x = ExTensor(np.linspace(-2.0, 2.0, 12).reshape(3, 4))
    cpu.reset_flops()

    getattr(x, name)()

    assert cpu.flop_count() == flops_per_element * x.data.size
