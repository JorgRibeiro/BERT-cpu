"""Minimal tokenizer for feeding text into the encoder."""

from __future__ import annotations

from typing import Dict, List, Optional


class Tokenizer:
    """A small whitespace/WordPiece-style tokenizer with special tokens.

    Handles the vocabulary, the BERT special tokens (``[PAD]``, ``[UNK]``,
    ``[CLS]``, ``[SEP]``, ``[MASK]``), and conversion between text and id
    sequences (with padding/truncation).
    """

    def __init__(self, vocab: Optional[Dict[str, int]] = None) -> None:
        raise NotImplementedError

    def build_vocab(self, corpus: List[str], max_size: int = 30000) -> None:
        """Build the vocabulary from a corpus of raw text."""
        raise NotImplementedError

    def tokenize(self, text: str) -> List[str]:
        """Split text into tokens."""
        raise NotImplementedError

    def encode(
        self, text: str, max_len: Optional[int] = None, add_special_tokens: bool = True
    ) -> List[int]:
        """Convert text into a list of token ids."""
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        """Convert a list of token ids back into text."""
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError
