"""Tests for the autograd engine.

This file has two layers:

1. **Software tests** (``test_*`` functions): ordinary correctness checks. The
   most valuable one is a numerical gradient check (compare analytic gradients
   from ``backward`` against finite differences), since broadcasting in the
   backward pass is the easiest thing to get subtly wrong.

2. **A didactic visualisation** (the ``draw_graph`` / ``draw_backward_steps``
   helpers and the ``demo_gradient_graph`` walkthrough): these print an input
   vector and an ASCII drawing of the computational graph, annotating every
   node with its forward value *and* its gradient so a reader can literally see
   how reverse-mode autodiff propagates the chain rule from the output back to
   the inputs. The demo computes ``y = tanh(wᵀ @ x)`` — a tiny linear layer —
   using the "bias trick" (a leading ``x_0 = 1`` so the bias is just another
   weight ``w_0``).

Run the demo on its own with::

    python test/test_engine.py

or see the same output during the test run with::

    pytest -s test/test_engine.py -k demo
"""

import numpy as np

import bert_cpu.engine as engine
from bert_cpu.engine import Tensor
from test.console import console


# ====================================================================== #
# Helpers
# ====================================================================== #
def numeric_gradient(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Estimate the gradient of a scalar function ``f()`` w.r.t. ``x`` numerically.

    Why this exists
    ---------------
    This is the independent *reference* against which we check the gradients
    computed by our autograd engine, i.e., by ``Tensor.backward()``. In this
    library, ``Tensor.backward()`` traverses the computational graph in reverse
    topological order and calls each node's local ``_backward`` closure to apply
    the chain rule.

    The trick of a *gradient check* is to compute the same gradient a second way,
    using a method that knows nothing about the engine's internals, and compare.
    If the numerical and analytic gradients agree within tolerance, we gain
    confidence that ``Tensor.backward()`` and the local ``_backward`` rules are
    implemented correctly. If they disagree, then some part of the backward pass
    is likely wrong: either a local ``_backward`` rule, gradient accumulation, or
    the chain-rule traversal. (See ``test_gradcheck_mlp`` and
    ``test_softmax_sums_to_one``.)

    What exactly is ``f``?
    ----------------------
    ``f`` is the function we are differentiating: it re-runs the forward pass
    and returns the single scalar output as a plain Python float. Crucially it
    takes **no arguments** -- it does not receive ``x``. Instead it reads ``x``
    *indirectly*, by closing over the very same array object that this routine
    mutates. So ``f`` and ``x`` are wired to the same memory: when we poke
    ``x[i]``, the next call to ``f()`` recomputes the output using that poked
    value. That shared-array link is the whole mechanism -- it is how perturbing
    one number changes the output we measure.

    The maths
    ---------
    The gradient is the vector of partial derivatives, and each partial is just
    a derivative -- the limit of a difference quotient from first-year calculus:

        df/dx_i = lim_{h->0} ( f(x + h*e_i) - f(x) ) / h

    where ``e_i`` is the unit step along coordinate ``i``. We cannot take the
    limit numerically, so we approximate it with a small finite ``h = eps``.
    Instead of the one-sided quotient above we use the **central difference**,
    which perturbs ``x_i`` both up and down::

        df/dx_i ≈ ( f(x + eps*e_i) - f(x - eps*e_i) ) / (2 * eps)

    The central form is used because its error shrinks as O(eps^2) (the first
    error term cancels by symmetry), whereas the one-sided form is only O(eps),
    so for the same ``eps`` the central estimate is far more accurate.

    How it is computed
    ------------------
    There is no analytic shortcut here: we literally evaluate ``f`` twice for
    every single element of ``x`` (perturbed up, then down), filling the
    gradient entry by entry. That is why this is O(size of ``x``) forward passes
    and only ever used on tiny tensors inside tests -- never to actually train.



    Concretely, a caller writes (see ``test_gradcheck_mlp``)::

        x = Tensor(np.random.randn(4, 3))          # a leaf in the graph
        def forward():
            return ((x @ W + b).gelu() ** 2).mean()  # rebuilds graph, scalar out
        numeric_gradient(lambda: float(forward().data), x.data)

    Here ``f = lambda: float(forward().data)`` rebuilds the expression and
    returns the loss, and the ``x`` we differentiate is ``x.data`` -- the *same*
    NumPy buffer the Tensor ``x`` holds. Because ``forward`` reads ``x`` (and
    thus ``x.data``) every time it runs, editing ``x.data[i]`` in place is seen
    by the next ``forward()``. (Pass a *copy* of the array and the link breaks:
    ``f`` would keep reading the original and every partial would come out 0.)

    Parameters
    ----------
    f : Callable[[], float]
        Zero-argument closure that re-evaluates the forward pass over the
        *current* contents of ``x`` and returns the scalar output as a float.
    x : np.ndarray
        The exact array object that ``f`` reads (e.g. a Tensor's ``.data``), not
        a copy. It is perturbed element by element and always restored to its
        original value before returning.
    eps : float
        The finite step ``h``. A trade-off: too large and the difference
        quotient is a poor approximation of the limit (truncation error); too
        small and floating-point round-off dominates. ``1e-6`` is a good middle
        ground for ``float64``.

    Returns
    -------
    np.ndarray
        An array of the same shape as ``x`` holding the estimated gradient.
        Compared against the engine's ``.grad`` with a tolerance (e.g.
        ``atol=1e-5``), since finite differences are never bit-exact.
    """
    grad = np.zeros_like(x)
    # Iterate over every element of x, tracking its multi-dimensional index.
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        original = x[i]

        x[i] = original + eps        # perturb coordinate i up   -> f(x + eps*e_i)
        plus = f()
        x[i] = original - eps        # perturb coordinate i down -> f(x - eps*e_i)
        minus = f()
        x[i] = original              # restore so the next element starts clean

        # Central difference: the slope of f along coordinate i at x.
        grad[i] = (plus - minus) / (2.0 * eps)
        it.iternext()
    return grad


# ====================================================================== #
# Software tests
# ====================================================================== #
def test_add_broadcast_backward():
    """Adding a (d,) bias to a (n, d) tensor should sum grad over the batch."""
    n, d = 4, 3
    x = Tensor(np.random.randn(n, d))
    b = Tensor(np.random.randn(d))

    (x + b).sum().backward()

    # Every element of x feeds the sum once -> all-ones gradient.
    assert np.allclose(x.grad, np.ones((n, d)))
    # The bias is broadcast across all n rows, so its grad sums over the batch.
    assert np.allclose(b.grad, np.full(d, float(n)))


def test_matmul_backward():
    """matmul backward should match grad @ b.T and a.T @ grad."""
    a = Tensor(np.random.randn(4, 3))
    b = Tensor(np.random.randn(3, 5))

    out = a @ b
    out.backward()  # seeds out.grad with ones of shape (4, 5)

    upstream = np.ones((4, 5))
    assert np.allclose(a.grad, upstream @ b.data.T)
    assert np.allclose(b.grad, a.data.T @ upstream)


def test_softmax_sums_to_one():
    """Softmax output is a valid probability distribution along its axis."""
    logits = Tensor(np.random.randn(6, 10))
    probs = logits.softmax(axis=-1)

    assert np.allclose(probs.data.sum(axis=-1), 1.0)
    assert np.all(probs.data >= 0.0)

    # And its Jacobian is correct: gradient check against finite differences.
    # A fixed weight matrix keeps the scalar objective deterministic so that
    # the analytic and numeric passes measure the same function.
    coeff = np.random.randn(6, 10)

    def f():
        x = logits.data - logits.data.max(axis=-1, keepdims=True)
        e = np.exp(x)
        sm = e / e.sum(axis=-1, keepdims=True)
        return float((sm * coeff).sum())

    logits.zero_grad()
    (logits.softmax(axis=-1) * Tensor(coeff)).sum().backward()
    analytic = logits.grad.copy()
    numeric = numeric_gradient(f, logits.data)
    assert np.allclose(analytic, numeric, atol=1e-5)


def test_gradcheck_mlp():
    """Finite-difference gradient check of a small composite expression.

    Computes ``loss = mean(gelu(x @ W + b) ** 2)`` and verifies the analytic
    gradients w.r.t. every input against central finite differences.
    """
    np.random.seed(0)
    x = Tensor(np.random.randn(4, 3))
    W = Tensor(np.random.randn(3, 5))
    b = Tensor(np.random.randn(5))

    def forward():
        return ((x @ W + b).gelu() ** 2).mean()

    for t in (x, W, b):
        t.zero_grad()
    forward().backward()

    for name, t in (("x", x), ("W", W), ("b", b)):
        numeric = numeric_gradient(lambda: float(forward().data), t.data)
        assert np.allclose(t.grad, numeric, atol=1e-5), f"gradient mismatch for {name}"


# ====================================================================== #
# Didactic visualisation of the gradient graph
# ====================================================================== #
def draw_graph(root: Tensor, known: set) -> None:
    """Print the computational graph as an aligned ``name | value | grad`` table.

    One row per node, ordered from the output (top) down to the inputs
    (bottom). The ``name``, ``value`` and ``grad`` columns are each aligned
    vertically. Gradients of nodes not in ``known`` print as ``-`` (not computed
    yet), so redrawing the table with a growing ``known`` set animates how
    ``backward`` fills the gradients in.
    """
    # Nodes top-to-bottom: output first, then its inputs (depth-first).
    order: list = []
    seen: set = set()

    def visit(node: Tensor) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        order.append(node)
        children = sorted(
            node._prev,
            key=lambda c: (getattr(c, "label", "") or c._op or "", id(c)),
        )
        for child in children:
            visit(child)

    visit(root)

    # Build the plain (uncoloured) cell text so widths can be measured.
    rows = []
    for node in order:
        name = getattr(node, "label", None) or (node._op or "leaf")
        value = console.fmt_plain(node.data)
        grad = console.fmt_plain(node.grad) if id(node) in known else "-"
        rows.append((name, value, grad))

    name_w = max(len("name"), *(len(r[0]) for r in rows))
    value_w = max(len("value"), *(len(r[1]) for r in rows))
    grad_w = max(len("grad"), *(len(r[2]) for r in rows))

    indent = " " * 10 # push the whole table to the right

    def line(name_cell, value_cell, grad_cell) -> str:
        return f"{indent}{name_cell}  |  {value_cell}  |  {grad_cell}"

    # Blank line, header + separator, one coloured row per node, blank line.
    print()
    print(line(
        console.label("name".ljust(name_w)),
        console.label("value".ljust(value_w)),
        console.label("grad".ljust(grad_w)),
    ))
    print(f"{indent}{'-' * name_w}--+--{'-' * value_w}--+--{'-' * grad_w}")
    for name, value, grad in rows:
        grad_cell = console.value(grad.ljust(grad_w)) if grad != "-" else grad.ljust(grad_w)
        print(line(
            console.math(name.ljust(name_w)),
            console.value(value.ljust(value_w)),
            grad_cell,
        ))
    print()


def _T(name: str) -> str:
    """Toggle a trailing transpose marker ``ᵀ`` on a variable name.

    So ``_T("x") == "xᵀ"`` and ``_T("wᵀ") == "w"`` (a transpose cancels itself),
    which keeps the matmul backward formulas readable.
    """
    return name[:-1] if name.endswith("ᵀ") else name + "ᵀ"


def _backward_rule(v: Tensor) -> "tuple[str, str]":
    """Return ``(formula, note)`` for how node ``v``'s backward feeds its inputs.

    Spells out the local derivative used by ``v._backward`` so the reader sees
    exactly how each child gradient is computed (the chain-rule step). The
    children are taken in the same sorted order ``draw_graph`` uses. For matmul
    this assumes that sorted order matches the engine's ``self @ other`` operand
    order, which holds for the demo's ``wᵀ @ x``.
    """
    vname = getattr(v, "label", "") or v._op or "?"
    kids = [
        getattr(c, "label", "") or c._op or "?"
        for c in sorted(
            v._prev, key=lambda c: (getattr(c, "label", "") or c._op or "", id(c))
        )
    ]

    if v._op == "tanh" and len(kids) == 1:
        z = kids[0]
        return (f"grad({z}) = grad({vname}) * (1 - {vname}^2)",
                f"local derivative of tanh: d{vname}/d{z} = 1 - {vname}^2")
    if v._op == "@" and len(kids) == 2:
        a, b = kids  # for the demo: a = wᵀ, b = x
        return (f"grad({a}) = grad({vname}) @ {_T(b)},   grad({b}) = {_T(a)} @ grad({vname})",
                "matmul backward: upstream grad times the *other* factor, transposed")
    if v._op == "transpose" and len(kids) == 1:
        c = kids[0]
        return (f"grad({c}) = grad({vname})ᵀ",
                "transpose just transposes the gradient straight back")
    if v._op == "sum" and len(kids) == 1:
        c = kids[0]
        return (f"grad({c}) = grad({vname}) copied into every element of {c}",
                "sum sends the same upstream gradient to each summand")
    if v._op == "*" and len(kids) == 2:
        a, b = kids
        return (f"grad({a}) = grad({vname}) * {b},   grad({b}) = grad({vname}) * {a}",
                "elementwise product rule: each factor's grad is the upstream grad "
                "times the other factor")
    # Generic fallback for any other op.
    return (f"grad(child) = grad({vname}) * d{vname}/dchild", "chain rule")


def draw_backward_steps(root: Tensor) -> None:
    """Run backprop one node at a time, redrawing the graph after each reveal.

    Replays exactly what ``Tensor.backward`` does (topological order, seed the
    output, then call each node's ``_backward`` in reverse), but pauses after
    every step to redraw the graph. A node's gradient is only revealed once
    *all* of its consumers have run their ``_backward`` (so the accumulation is
    complete); until then it prints as ``-``.
    """
    # Topological order: children before parents (same as Tensor.backward).
    topo: list = []
    visited: set = set()

    def build(v: Tensor) -> None:
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)

    build(root)

    # For each node, the set of consumers that feed gradient back into it.
    consumers = {id(n): set() for n in topo}
    for n in topo:
        for child in n._prev:
            consumers[id(child)].add(id(n))

    # Fresh start: zero every gradient, then seed the output with dy/dy = 1.
    for n in topo:
        n.grad = np.zeros_like(n.data)
    root.grad = np.ones_like(root.data)
    known = {id(root)}

    step = 0
    print(console.label(f"\nSTEP {step}") + console.text(": seed the output, ")
          + console.math("grad(y) = dy/dy = 1") + console.text("; everything else is '-'"))
    draw_graph(root, known)

    processed: set = set()
    for v in reversed(topo):
        v._backward()
        processed.add(id(v))
        # A child is now fully known once every consumer of it is processed.
        newly = [
            c for c in v._prev
            if id(c) not in known and consumers[id(c)] <= processed
        ]
        if not newly:
            continue
        for c in newly:
            known.add(id(c))
        step += 1
        vname = getattr(v, "label", "") or v._op or "leaf"
        filled = ", ".join(getattr(c, "label", "") or c._op or "leaf" for c in newly)
        formula, note = _backward_rule(v)
        print(console.label(f"\nSTEP {step}") + console.text(": backprop through ") + console.math(vname)
              + console.text(" fills the gradient of ") + console.math(filled))
        print(console.text("        ") + console.math(formula) + console.text(f"\n        [{note}]"))
        draw_graph(root, known)



def demo_gradient_graph() -> None:
    """Build a tiny linear unit, run autodiff, and visualise the gradient graph.

    The unit computes ``y = tanh(wᵀ @ x)`` — a single matrix multiplication (a
    linear layer) followed by a tanh. The "bias trick" augments the input with a
    leading constant ``x_0 = 1`` so the bias is just another weight ``w_0``, and
    ``x`` and ``w`` are column vectors so ``wᵀ @ x`` is their weighted sum.
    """
    print('\n')
    print(console.text("=" * 64))
    print(
        console.text("Gradient-graph demo:  ")
        + console.math("y = tanh( wᵀ @ x )")
        + console.text(",  with ")
        + console.math("x_0 = 1")
    )
    print(console.text("=" * 64))

    # Real features of the input (random each run; seed via engine.set_seed).
    features = np.random.randn(2)
    # Augmented input column vector x = [x_0=1, x_1, x_2]ᵀ, shape (3, 1).
    x = Tensor([[1.0], [features[0]], [features[1]]]);   x.label = "x"
    # Weight column vector w = [w_0(bias), w_1, w_2]ᵀ, shape (3, 1). Kept small
    # (* 0.3) so the pre-activation stays in tanh's active region and the
    # gradients don't vanish into a saturated tanh.
    w = Tensor(np.random.randn(3, 1) * 0.3);             w.label = "w"

    print(console.text("\nReal features      : "), end="")
    print(console.fmt(features))
    print(console.text("Augmented input  ") + console.math("x = "), end="")
    print(console.fmt(x.data), end="")
    print(console.text("    (column (3,1); ") + console.math("x_0 = 1") + console.text(" prepended)"))
    print(console.text("Weights          ") + console.math("w = "), end="")
    print(console.fmt(w.data), end="")
    print(console.text("    (column (3,1); ") + console.math("w_0") + console.text(" is the bias)"))

    # Forward pass: transpose w, then the matmul (linear layer), then tanh.
    wt = w.T;       wt.label = "wᵀ"   # transpose -> row (1, 3)
    z = wt @ x;     z.label = "z"     # (1, 3) @ (3, 1) -> (1, 1)
    y = z.tanh();   y.label = "y"

    print(console.text("\nForward pass:"))
    console.kv("  wᵀ           = ", console.fmt(wt.data), "    (w transposed -> row)", color=console.math)
    console.kv("  z = wᵀ @ x   = ", console.fmt(z.data), "    (matrix product = weighted sum + bias)", color=console.math)
    console.kv("  y = tanh(z)  = ", console.fmt(y.data), color=console.math)

    print(console.text("\nEach node in the computational graph stores:"))
    print(
        console.text("a ") + console.label("name") + console.text(" a forward ")
        + console.label("value") + console.text(" and a ") + console.label("grad") + console.text(" = ") + console.math("d(output)/d(node)")
    )
    print(console.text("allowing the system to track both the forward computation\nand the backward flow of derivatives:\n"))

    draw_graph(y, known=set())
   

    print(console.text("\nBackprop begins by seeding the output, ") 
        + console.math("dy/dy = 1") 
        + console.text("."))

    print(console.text("(In regular training, this initial value would be ")
        + console.math("dL/dy")
        + console.text(", where L \nis a loss function; here, however, we are differentiating ")
        + console.math("y")
        + console.text(" itself.)"))

    print(console.math("dy/dy")
        + console.text(" is then used to compute ")
        + console.math("dy/dz = grad(z)")
        + console.text("."))

    # Replay backprop step by step, redrawing the graph as each grad is filled.
    draw_backward_steps(y)

def test_demo_gradient_graph_runs():
    """Smoke test: the didactic walkthrough runs and its assertions hold.

    Run ``pytest -s -k demo`` to actually see the printed graph.
    """
    demo_gradient_graph()


# ====================================================================== #
# Standalone entry point
# ====================================================================== #
def main(argv=None) -> None:
    """Run the didactic walkthrough with a chosen precision and RNG seed.

    The precision and seed are configured on the engine *before* anything is
    built, so they govern the whole run::

        python test/test_engine.py --precision float32 --seed 0
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision",
        default="float64",
        help="NumPy float dtype for the engine (e.g. float16, float32, float64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the NumPy RNG (reproducible run). Omit for a random run.",
    )
    args = parser.parse_args(argv)

    # Resolve the precision string to a NumPy dtype (clear error if invalid).
    try:
        dtype = np.dtype(args.precision)
    except TypeError as exc:
        parser.error(f"unknown precision {args.precision!r}: {exc}")
    if not np.issubdtype(dtype, np.floating):
        parser.error(f"precision must be a floating type, got {args.precision!r}")

    # Configure the engine globally, before any tensor is created. Only seed
    # when asked, so an unseeded run draws fresh random tensors each time.
    engine.default_dtype = dtype.type
    if args.seed is not None:
        engine.set_seed(args.seed)

    seed_str = "random" if args.seed is None else str(args.seed)
    print(console.text("Engine config: ") + console.label("seed") + console.text("=")
          + console.value(seed_str) + console.text("  ") + console.label("precision")
          + console.text("=") + console.value(dtype.name))
    demo_gradient_graph()


if __name__ == "__main__":
    main()
