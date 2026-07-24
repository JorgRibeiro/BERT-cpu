from setuptools import setup, find_packages

setup(
    name="bert-cpu",
    version="0.1.0",
    author="Luis",
    description=(
        "A tiny NumPy-backed BERT-style Transformer encoder, "
        "a tensor-valued successor to micrograd."
    ),
    packages=find_packages(),
    install_requires=[
        "numpy>=2.1,<2.2",
        "matplotlib>=3.9,<3.10",
    ],
    extras_require={"test": ["pytest>=8.3,<8.4"]},
    python_requires=">=3.10",
)
