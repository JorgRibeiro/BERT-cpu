"""Focused tests for the Variable 3 affine-only Adult models."""

from __future__ import annotations

import numpy as np
import pytest

from bert_cpu import engine as cpu
from exercises import task_binary_classification as adult
from experiments import run_v1


EXPECTED_PARAMETERS = {
    1: 218,
    2: 7_106,
    3: 11_266,
}
EXPECTED_WEIGHT_SHAPES = {
    1: [(109, 2)],
    2: [(109, 64), (65, 2)],
    3: [(109, 64), (65, 64), (65, 2)],
}


def _graph_ops(output: cpu.Tensor) -> list[str]:
    seen: set[int] = set()
    operations: list[str] = []

    def visit(node: cpu.Tensor) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        operations.append(node._op)
        for parent in node._prev:
            visit(parent)

    visit(output)
    return operations


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_linear_depth_shape_parameters_gradients_and_no_activation(depth):
    rng = np.random.RandomState(3)
    X = rng.normal(size=(108, 7))
    model = adult.build_linear_model(108, depth=depth, model_seed=1)

    logits = model(cpu.Tensor(X, requires_grad=False))

    assert logits.shape == (2, 7)
    assert model.layer_sizes[0] == 108
    assert model.layer_sizes[-1] == 2
    assert adult.parameter_count(model) == EXPECTED_PARAMETERS[depth]
    assert [name for name, _ in model.named_parameters()] == [
        f"layers.{index}.weight" for index in range(depth)
    ]
    assert [parameter.shape for parameter in model.parameters()] == (
        EXPECTED_WEIGHT_SHAPES[depth]
    )
    assert _graph_ops(logits).count("@") == depth
    assert set(_graph_ops(logits)).isdisjoint(
        {"relu", "sigmoid", "swish", "softplus", "softplus_beta", "identity"}
    )

    logits.sum().backward()
    assert all(np.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert all(np.any(parameter.grad != 0.0) for parameter in model.parameters())


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_composed_model_matches_one_collapsed_affine_function(depth):
    rng = np.random.RandomState(11)
    X = rng.normal(size=(108, 13))
    model = adult.build_linear_model(108, depth=depth, model_seed=2)

    matrix, bias = adult.collapse_affine(model)
    expected = matrix @ X + bias[:, None]
    actual = model(cpu.Tensor(X, requires_grad=False)).data

    assert matrix.shape == (2, 108)
    assert bias.shape == (2,)
    assert np.linalg.matrix_rank(matrix) <= 2
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_same_seed_preserves_shared_first_layer_between_l2_and_l3():
    l2 = adult.build_linear_model(108, depth=2, model_seed=7)
    l3 = adult.build_linear_model(108, depth=3, model_seed=7)

    np.testing.assert_array_equal(
        l2.layers[0].weight.data,
        l3.layers[0].weight.data,
    )


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_l2_initial_weights_are_exactly_the_relu_linear_weights(seed):
    linear = adult.build_linear_model(108, depth=2, model_seed=seed)
    relu = adult.build_model(108, activation="relu", model_seed=seed)

    assert run_v1.parameter_hash(linear) == run_v1.parameter_hash(relu)
    for linear_parameter, relu_parameter in zip(
        linear.parameters(),
        relu.parameters(),
    ):
        np.testing.assert_array_equal(linear_parameter.data, relu_parameter.data)


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_linear_builder_is_reproducible_for_each_depth(depth):
    first = adult.build_linear_model(108, depth=depth, model_seed=9)
    second = adult.build_linear_model(108, depth=depth, model_seed=9)

    assert run_v1.parameter_hash(first) == run_v1.parameter_hash(second)
    for left, right in zip(first.parameters(), second.parameters()):
        np.testing.assert_array_equal(left.data, right.data)


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_forward_flops_contain_only_the_configured_matmuls(depth):
    batch = 7
    model = adult.build_linear_model(108, depth=depth, model_seed=0)
    X = np.zeros((108, batch))
    expected = sum(
        2 * output_size * batch * (input_size + 1)
        for input_size, output_size in zip(
            model.layer_sizes,
            model.layer_sizes[1:],
        )
    )

    cpu.reset_flops()
    model(cpu.Tensor(X, requires_grad=False))

    assert cpu.flop_count() == expected


@pytest.mark.parametrize("invalid", (0, 4, -1, 1.5, True, "2"))
def test_invalid_linear_depth_is_rejected(invalid):
    with pytest.raises(ValueError, match="depth"):
        adult.AdultLinearClassifier(108, depth=invalid)


@pytest.mark.parametrize(
    ("n_features", "hidden"),
    ((0, 64), (108, 0), (-1, 64), (108, -1), (1.5, 64), (108, True)),
)
def test_invalid_linear_dimensions_are_rejected(n_features, hidden):
    with pytest.raises(ValueError, match="positive integers"):
        adult.AdultLinearClassifier(n_features, depth=2, hidden=hidden)


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_collapse_affine_does_not_change_weights_rng_or_flops(depth):
    model = adult.build_linear_model(108, depth=depth, model_seed=4)
    weights_before = [parameter.data.copy() for parameter in model.parameters()]
    rng_before = np.random.get_state()
    cpu.reset_flops()

    matrix, bias = adult.collapse_affine(model)

    assert matrix.dtype == model.parameters()[0].data.dtype
    assert bias.dtype == model.parameters()[0].data.dtype
    assert cpu.flop_count() == 0
    for expected, parameter in zip(weights_before, model.parameters()):
        np.testing.assert_array_equal(parameter.data, expected)
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_one_training_step_is_finite_for_each_linear_depth(depth):
    rng = np.random.RandomState(5)
    X = rng.normal(size=(8, 30))
    y = np.array([0, 1] * 15)
    X_tr, y_tr, X_val, y_val = adult.train_val_split(X, y, split_seed=0)
    model = adult.build_linear_model(8, depth=depth, model_seed=0)

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
        [
            metrics.train_loss,
            metrics.val_loss,
            metrics.train_accuracy,
            metrics.val_accuracy,
        ]
    ).all()
    assert all(np.isfinite(parameter.data).all() for parameter in model.parameters())
