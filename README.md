# BERT-cpu

An educational, from-scratch implementation of BERT using only NumPy, built to run on CPU.

## Requirements

- Python >= 3.8
- NumPy (the only runtime dependency)

## Setup

Create a virtual environment and install the dependencies:

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows (PowerShell)

# Install the dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** if `python3 -m venv` fails with an `ensurepip is not available`
> error, your interpreter is missing the `venv` module. On Debian/Ubuntu install
> it with `apt install python3-venv`, or use a `pyenv`-managed Python.

## Learning path (start here)

This project is meant to be *read and run* from the ground up. Everything BERT
does eventually reduces to one idea: a graph of tensor operations through which
gradients flow backward. So the very first thing to understand — right after
installing — is **how the autograd engine works**.

**Step 1 — see the gradient engine in action.** With the virtual environment
activated, run *only* the engine's didactic demo:

```bash
pip install pytest                                          # if not done yet
pytest -s -k demo test/test_engine.py
```

- `-s` lets the demo print to your screen (pytest hides output otherwise).
- `-k demo` selects only the walkthrough test (`test_demo_gradient_graph_runs`).
- `test/test_engine.py` restricts the run to the engine file.

You can get the same output as a plain script:

```bash
python test/test_engine.py
```

What you will see, and what to take away from it:

1. **An input vector** `x` and a **weight vector** `w`. The demo uses the
   "bias trick": the input is *augmented* with a leading constant `x_0 = 1`, so
   `w` has one more component than there are real features and `w_0` plays the
   role of the bias.
2. **The forward pass**, computing the inner product `z = w . x` and then
   `y = tanh(z)` step by step.
3. **An ASCII drawing of the computational graph**, output on top and inputs at
   the leaves. Each node is annotated with both its forward value *and* its
   gradient.
4. **The chain rule, by hand**, printed next to the gradients autograd produced
   — so you can confirm the engine is just calculus, mechanised.

Read the graph from the top down to follow the forward pass, then read the
gradients to see how `backward()` distributes the chain rule from the output
back to every input. Once that "click" happens, the rest of the library
(layers, attention, the full encoder) is just *bigger graphs of the same kind*.

**Step 2 — confirm the engine is correct.** Run the engine's software tests
(broadcasting, matmul, softmax, finite-difference gradient checks):

```bash
pytest test/test_engine.py
```

From here you are ready to explore the higher-level modules. The full testing
reference is in [Tests and didactic walkthroughs](#tests-and-didactic-walkthroughs).

## Usage

With the virtual environment activated, import the library from the project
root. The package directory is `bert_cpu` (underscore — the importable name),
while the distribution is named `bert-cpu`:

```python
from bert_cpu import BERTModel, Adam, cross_entropy
from bert_cpu.transformer import EncoderLayer

model = BERTModel(vocab_size=30000)
```

## Tests and didactic walkthroughs

The test suite serves two purposes, kept side by side in the same files:

1. **Software tests** — ordinary correctness checks (broadcasting, matmul,
   softmax, finite-difference gradient checks, etc.).
2. **Didactic walkthroughs** — runnable demos that *print to the console* to
   teach what is happening internally, e.g. drawing the computational graph and
   showing how reverse-mode autodiff propagates the chain rule from the output
   back to the inputs.

First install the test dependency (already covered if you ran the setup above):

```bash
pip install pytest
```

### Run the software tests

```bash
pytest                       # run everything quietly
pytest -v                    # verbose, one line per test
pytest test/test_engine.py   # just the autograd-engine tests
```

### Run the didactic walkthroughs

The walkthroughs print explanatory output, so run them with `-s` (so pytest does
not capture stdout) and select them with `-k demo`:

```bash
pytest -s -k demo                       # every didactic walkthrough
pytest -s -k demo test/test_engine.py   # only the engine's gradient-graph demo
```

Each walkthrough is also runnable as a plain script for a clean, standalone
view. For example, the gradient-graph demo (input vector → forward pass →
ASCII graph annotated with gradients → chain-rule check):

```bash
python test/test_engine.py
```

> New here? Follow the [Learning path](#learning-path-start-here) above — it
> walks you through this engine demo first, since every other module is built
> on top of it.
