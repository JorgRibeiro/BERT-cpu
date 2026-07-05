"""Example datasets bundled with BERT-cpu.

Import this module and call a loader to get model-ready NumPy arrays::

    import datasets

    train = datasets.load_adult("train")
    test  = datasets.load_adult("test")

    train.X            # (n_features, n_samples) float array — column-oriented
    train.y            # (n_samples,) int labels in {0, 1}  (1 == income >50K)
    train.feature_names
    train.categories

The arrays are plain NumPy, so wrap them in the engine when you want to train::

    from bert_cpu import engine as cpu
    X = cpu.Tensor(train.X)            # already (features, batch)
"""

from datasets.loaders import AdultDataset, load_adult

__all__ = ["load_adult", "AdultDataset"]
