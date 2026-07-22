"""Focused tests for Adult integration of q01 Variable 1."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bert_cpu import engine as cpu
from exercises import task_binary_classification as adult


def _parameter_arrays(model: adult.AdultMLP) -> list[np.ndarray]:
    return [parameter.data.copy() for parameter in model.parameters()]


def test_default_is_exactly_explicit_relu_and_keeps_baseline_size():
    X = np.linspace(-1.0, 1.0, 108 * 5).reshape(108, 5)

    cpu.set_seed(7)
    default_model = adult.AdultMLP(108)
    cpu.set_seed(7)
    explicit_model = adult.AdultMLP(108, activation="relu")

    assert default_model.activation == "relu"
    assert adult.parameter_count(default_model) == 7_106
    assert default_model(cpu.Tensor(X, requires_grad=False)).shape == (2, 5)
    np.testing.assert_allclose(
        default_model(cpu.Tensor(X, requires_grad=False)).data,
        explicit_model(cpu.Tensor(X, requires_grad=False)).data,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("activation", adult.ACTIVATIONS)
def test_each_v1_activation_keeps_shape_and_fc1_connected(activation):
    X = np.linspace(-2.0, 2.0, 6 * 8).reshape(6, 8)
    model = adult.build_model(6, activation=activation, model_seed=3)

    logits = model(cpu.Tensor(X, requires_grad=False))
    assert logits.shape == (2, 8)
    assert adult.parameter_count(model) == (6 + 1) * 64 + (64 + 1) * 2
    assert np.isfinite(logits.data).all()

    logits.sum().backward()
    assert np.isfinite(model.fc1.weight.grad).all()
    assert np.any(model.fc1.weight.grad != 0.0), "activation detached fc1"


def test_model_seed_gives_identical_linear_weights_for_every_activation():
    reference = _parameter_arrays(adult.build_model(108, "relu", model_seed=1))

    for activation in adult.ACTIVATIONS[1:]:
        candidate = _parameter_arrays(adult.build_model(108, activation, model_seed=1))
        for expected, actual in zip(reference, candidate):
            np.testing.assert_array_equal(actual, expected)


def test_unknown_activation_is_rejected():
    with pytest.raises(ValueError, match="unknown activation"):
        adult.AdultMLP(4, activation="gelu")


def test_split_seed_zero_reproduces_historical_global_permutation():
    n_samples = 40
    val_frac = 0.2
    np.random.seed(0)
    historical = np.random.permutation(n_samples)

    train_idx, val_idx = adult.train_val_indices(
        n_samples,
        val_frac=val_frac,
        split_seed=0,
    )

    n_val = int(n_samples * val_frac)
    np.testing.assert_array_equal(val_idx, historical[:n_val])
    np.testing.assert_array_equal(train_idx, historical[n_val:])


def test_split_is_reproducible_without_consuming_model_randomness():
    first_train, first_val = adult.train_val_indices(50, split_seed=0)

    # Unrelated global draws cannot change the local split RNG.
    np.random.seed(999)
    np.random.normal(size=100)
    second_train, second_val = adult.train_val_indices(50, split_seed=0)
    np.testing.assert_array_equal(first_train, second_train)
    np.testing.assert_array_equal(first_val, second_val)

    # Applying the model seed immediately before construction restores exactly
    # the same two parameter tensors, regardless of split-related work.
    model_a = adult.build_model(5, model_seed=2)
    adult.train_val_indices(50, split_seed=17)
    model_b = adult.build_model(5, model_seed=2)
    for expected, actual in zip(_parameter_arrays(model_a), _parameter_arrays(model_b)):
        np.testing.assert_array_equal(actual, expected)


def test_train_returns_structured_history():
    rng = np.random.RandomState(4)
    X = rng.normal(size=(5, 20))
    y = np.array([0, 1] * 10)
    X_tr, y_tr, X_val, y_val = adult.train_val_split(X, y, split_seed=0)
    model = adult.build_model(5, activation="relu", model_seed=0)

    result = adult.train(
        model,
        X_tr,
        y_tr,
        X_val,
        y_val,
        epochs=2,
        verbose=False,
    )

    assert len(result.history) == 2
    assert result.total_flops == sum(item.flops for item in result.history)
    assert result.total_flops > 0
    assert result.final_train_accuracy == result.history[-1].train_accuracy
    assert result.final_val_accuracy == result.history[-1].val_accuracy
    for item in result.history:
        assert np.isfinite(
            [item.train_loss, item.val_loss, item.train_accuracy, item.val_accuracy]
        ).all()


def test_default_cli_flow_never_loads_official_test(monkeypatch):
    rng = np.random.RandomState(5)
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

    result = adult.main(["--epochs", "1", "--quiet"])

    assert loaded_splits == ["train"]
    assert result.test_accuracy is None
