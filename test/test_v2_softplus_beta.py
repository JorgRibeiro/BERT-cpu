"""Focused tests for Variable 2: fixed-curvature Softplus."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bert_cpu import engine as cpu
from exercises import task_binary_classification as adult
from exercises.q01_activations import ExTensor
from test.test_engine import numeric_gradient


BETAS = (0.5, 1.0, 2.0, 5.0)


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    non_negative = values >= 0.0
    result[non_negative] = 1.0 / (1.0 + np.exp(-values[non_negative]))
    exp_values = np.exp(values[~non_negative])
    result[~non_negative] = exp_values / (1.0 + exp_values)
    return result


def _reference(values: np.ndarray, beta: float) -> np.ndarray:
    return np.logaddexp(0.0, beta * values) / beta


@pytest.mark.parametrize("beta", BETAS)
def test_softplus_beta_values_shape_dtype_and_graph(beta):
    values = np.array([[-3.0, -0.5, 0.0], [0.5, 2.0, 5.0]], dtype=np.float32)
    x = ExTensor(values)

    out = x.softplus_beta(beta)

    np.testing.assert_allclose(out.data, _reference(values, beta), rtol=1e-6, atol=1e-7)
    assert out.shape == x.shape
    assert out.data.dtype == values.dtype
    assert out.requires_grad is True
    assert out._op == "softplus_beta"
    assert out._prev == {x}


@pytest.mark.parametrize("beta", BETAS)
def test_softplus_beta_extremes_and_derivatives_are_finite(beta):
    values = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0])
    x = ExTensor(values)

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        out = x.softplus_beta(beta)
        out.sum().backward()

    assert np.isfinite(out.data).all()
    assert np.isfinite(x.grad).all()
    np.testing.assert_allclose(out.data, _reference(values, beta))
    np.testing.assert_allclose(x.grad, _stable_sigmoid(beta * values))


@pytest.mark.parametrize("beta", BETAS)
def test_softplus_beta_gradient_matches_central_differences(beta):
    values = np.array([-2.3, -0.7, 0.2, 1.8])
    weights = np.array([0.5, -1.2, 2.0, 0.7])
    x = ExTensor(values)

    (x.softplus_beta(beta) * weights).sum().backward()
    numerical = numeric_gradient(
        lambda: float((_reference(x.data, beta) * weights).sum()),
        x.data,
    )

    np.testing.assert_allclose(x.grad, numerical, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("beta", BETAS)
def test_softplus_beta_unbound_method_preserves_base_graph(beta):
    values = np.array([-1.5, -0.1, 0.6, 2.0])
    weights = np.array([1.0, -0.25, 0.5, 1.5])
    x = cpu.Tensor(values)
    z = x * 1.7 + 0.2

    activated = ExTensor.softplus_beta(z, beta)
    (activated * weights).sum().backward()
    numerical = numeric_gradient(
        lambda: float((_reference(1.7 * x.data + 0.2, beta) * weights).sum()),
        x.data,
    )

    assert type(z) is cpu.Tensor
    assert z in activated._prev
    np.testing.assert_allclose(x.grad, numerical, rtol=1e-5, atol=1e-6)


def test_softplus_beta_accumulates_gradient_and_preserves_requires_grad():
    values = np.array([-1.0, 0.0, 2.0])
    x = ExTensor(values)
    activated = x.softplus_beta(2.0)
    (activated + activated).sum().backward()
    np.testing.assert_allclose(x.grad, 2.0 * _stable_sigmoid(2.0 * values))

    frozen = ExTensor(values, requires_grad=False)
    out = frozen.softplus_beta(2.0)
    out.sum().backward()
    assert out.requires_grad is False
    np.testing.assert_array_equal(frozen.grad, np.zeros_like(values))


@pytest.mark.parametrize(
    "beta",
    (0.0, -1.0, np.nan, np.inf, -np.inf, True, [1.0], 1.0 + 1.0j),
)
def test_softplus_beta_rejects_invalid_beta(beta):
    with pytest.raises(ValueError, match="finite positive scalar"):
        ExTensor(np.array([0.0])).softplus_beta(beta)


@pytest.mark.parametrize("beta", BETAS)
def test_softplus_beta_forward_cost_is_five_flops_per_element(beta):
    x = ExTensor(np.linspace(-2.0, 2.0, 12).reshape(3, 4))
    cpu.reset_flops()

    x.softplus_beta(beta)

    assert cpu.flop_count() == 5 * x.data.size


def test_beta_one_matches_softplus_values_and_gradients_but_not_flops():
    values = np.array([-1000.0, -2.0, 0.0, 3.0, 1000.0])
    standard_x = ExTensor(values.copy())
    beta_x = ExTensor(values.copy())

    cpu.reset_flops()
    standard = standard_x.softplus()
    standard_flops = cpu.flop_count()
    standard.sum().backward()

    cpu.reset_flops()
    beta_one = beta_x.softplus_beta(1.0)
    beta_flops = cpu.flop_count()
    beta_one.sum().backward()

    np.testing.assert_array_equal(beta_one.data, standard.data)
    np.testing.assert_array_equal(beta_x.grad, standard_x.grad)
    assert standard_flops == 3 * values.size
    assert beta_flops == 5 * values.size


def test_beta_mechanically_changes_offset_and_approaches_relu_on_fixed_grid():
    values = np.linspace(-8.0, 8.0, 1601)
    relu = np.maximum(0.0, values)
    distances = []

    for beta in BETAS:
        output = _reference(values, beta)
        zero_index = np.flatnonzero(values == 0.0).item()
        assert output[zero_index] == pytest.approx(np.log(2.0) / beta)
        distances.append(float(np.max(np.abs(output - relu))))

    assert distances[0] > distances[1] > distances[2] > distances[3]


@pytest.mark.parametrize("beta", BETAS)
def test_adult_softplus_beta_keeps_shape_parameters_and_fc1_gradient(beta):
    X = np.linspace(-2.0, 2.0, 6 * 8).reshape(6, 8)
    model = adult.build_model(
        6,
        activation="softplus_beta",
        model_seed=3,
        activation_beta=beta,
    )

    logits = model(cpu.Tensor(X, requires_grad=False))
    assert logits.shape == (2, 8)
    assert model.activation_beta == beta
    assert adult.parameter_count(model) == (6 + 1) * 64 + (64 + 1) * 2
    assert np.isfinite(logits.data).all()

    logits.sum().backward()
    assert np.isfinite(model.fc1.weight.grad).all()
    assert np.any(model.fc1.weight.grad != 0.0)


def test_beta_is_not_a_parameter_and_initial_weights_match_v1():
    relu = adult.build_model(108, activation="relu", model_seed=2)
    beta_model = adult.build_model(
        108,
        activation="softplus_beta",
        model_seed=2,
        activation_beta=5.0,
    )

    assert [name for name, _ in beta_model.named_parameters()] == [
        name for name, _ in relu.named_parameters()
    ]
    assert adult.parameter_count(beta_model) == 7_106
    for expected, actual in zip(relu.parameters(), beta_model.parameters()):
        np.testing.assert_array_equal(actual.data, expected.data)


def test_adult_rejects_missing_or_misplaced_beta():
    with pytest.raises(ValueError, match="requires activation_beta"):
        adult.AdultMLP(4, activation="softplus_beta")
    with pytest.raises(ValueError, match="only valid"):
        adult.AdultMLP(4, activation="relu", activation_beta=1.0)
    with pytest.raises(ValueError, match="finite positive scalar"):
        adult.AdultMLP(4, activation="softplus_beta", activation_beta=0.0)


def test_one_adam_epoch_is_finite_and_cost_is_identical_for_all_betas():
    rng = np.random.RandomState(8)
    X = rng.normal(size=(5, 20))
    y = np.array([0, 1] * 10)
    X_tr, y_tr, X_val, y_val = adult.train_val_split(X, y, split_seed=0)
    observed_flops = []

    for beta in BETAS:
        model = adult.build_model(
            5,
            activation="softplus_beta",
            model_seed=0,
            activation_beta=beta,
        )
        result = adult.train(
            model,
            X_tr,
            y_tr,
            X_val,
            y_val,
            epochs=1,
            verbose=False,
        )
        metrics = result.history[0]
        assert np.isfinite(
            [metrics.train_loss, metrics.val_loss, metrics.train_accuracy, metrics.val_accuracy]
        ).all()
        observed_flops.append(metrics.flops)

    assert len(set(observed_flops)) == 1


def test_v2_cli_never_loads_official_test_without_opt_in(monkeypatch):
    rng = np.random.RandomState(9)
    fake_train = SimpleNamespace(
        X=rng.normal(size=(4, 12)),
        y=np.array([0, 1] * 6),
        n_features=4,
    )
    loaded_splits = []

    def fake_load_adult(split):
        loaded_splits.append(split)
        if split != "train":
            raise AssertionError("official test was accessed without opt-in")
        return fake_train

    monkeypatch.setattr(adult.datasets, "load_adult", fake_load_adult)

    result = adult.main(
        [
            "--activation",
            "softplus_beta",
            "--activation-beta",
            "2",
            "--epochs",
            "1",
            "--quiet",
        ]
    )

    assert loaded_splits == ["train"]
    assert result.activation == "softplus_beta"
    assert result.activation_beta == 2.0
    assert result.test_accuracy is None
