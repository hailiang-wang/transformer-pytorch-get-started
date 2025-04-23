from typing import Dict, List, Optional

import torch
import torch.nn as nn


class Vocab(nn.Module):
    __jit_unused_properties__ = ["is_jitable"]
    r"""Creates a vocab object which maps tokens to indices.

    Args:
        vocab (torch.classes.torchtext.Vocab or torchtext._torchtext.Vocab): a cpp vocab object.
    """

    def __init__(self, tokens) -> None:
        super(Vocab, self).__init__()
        self.tokens = tokens
        self.__build()

    def __build(self) -> None:
        self.vocab_set = set()
        self.vocab_num2term = dict()
        self.vocab_term2num = dict()
        self.default_index = 0

        counter = 0

        for t in self.tokens:
            if not t in self.vocab_set:
                self.vocab_set.add(t)
                self.vocab_term2num[t] = counter
                self.vocab_num2term[counter] = t
                counter += 1

        self.vocab_size = len(self.vocab_set)

    @property
    def is_jitable(self):
        return isinstance(self.vocab, torch._C.ScriptObject)

    @torch.jit.export
    def forward(self, tokens: List[str]) -> List[int]:
        r"""Calls the `lookup_indices` method

        Args:
            tokens: a list of tokens used to lookup their corresponding `indices`.

        Returns:
            The indices associated with a list of `tokens`.
        """
        return self.lookup_indices(tokens)

    @torch.jit.export
    def __len__(self) -> int:
        r"""
        Returns:
            The length of the vocab.
        """
        return self.vocab_size

    @torch.jit.export
    def __contains__(self, token: str) -> bool:
        r"""
        Args:
            token: The token for which to check the membership.

        Returns:
            Whether the token is member of vocab or not.
        """
        return token in self.vocab_set

    @torch.jit.export
    def __getitem__(self, token: str) -> int:
        r"""
        Args:
            token: The token used to lookup the corresponding index.

        Returns:
            The index corresponding to the associated token.
        """
        return self.vocab_term2num[token]

    @torch.jit.export
    def set_default_index(self, index: Optional[int]) -> None:
        r"""
        Args:
            index: Value of default index. This index will be returned when OOV token is queried.
        """
        self.default_index = index

    @torch.jit.export
    def get_default_index(self) -> Optional[int]:
        r"""
        Returns:
            Value of default index if it is set.
        """
        return self.default_index

    @torch.jit.export
    def insert_token(self, token: str, index: int) -> None:
        r"""
        Args:
            token: The token used to lookup the corresponding index.
            index: The index corresponding to the associated token.
        Raises:
            RuntimeError: If `index` is not in range [0, Vocab.size()] or if `token` already exists in the vocab.
        """

        if token in self.vocab_set:
            raise RuntimeError("`token` already exists in the vocab.")
        
        if index < 0 or index > self.vocab_size:
            raise RuntimeError("`index` is not in range [0, Vocab.size()]")

        self.tokens.insert(index, token)
        self.__build()

    @torch.jit.export
    def append_token(self, token: str) -> None:
        r"""
        Args:
            token: The token used to lookup the corresponding index.

        Raises:
            RuntimeError: If `token` already exists in the vocab
        """
        if token in self.vocab_set:
            raise RuntimeError("`token` already exists in the vocab.")
        
        self.tokens.append(token)
        idx = self.vocab_size
        self.vocab_num2term[idx] = token
        self.vocab_term2num[token] = idx
        self.vocab_set.add(token)
        self.vocab_size += 1

    @torch.jit.export
    def lookup_token(self, index: int) -> str:
        r"""
        Args:
            index: The index corresponding to the associated token.

        Returns:
            token: The token used to lookup the corresponding index.

        Raises:
            RuntimeError: If `index` not in range [0, itos.size()).
        """
        if index >= 0 and index < self.vocab_size:
            return self.vocab_num2term[index]
        else:
            raise RuntimeError("`index` not in range [0, itos.size())")

    @torch.jit.export
    def lookup_tokens(self, indices: List[int]) -> List[str]:
        r"""
        Args:
            indices: The `indices` used to lookup their corresponding`tokens`.

        Returns:
            The `tokens` associated with `indices`.

        Raises:
            RuntimeError: If an index within `indices` is not int range [0, itos.size()).
        """

        ret = []

        for num in indices:
            if num >= 0 and num < self.vocab_size:
                ret.append(self.vocab_num2term[num])
            else:
                raise RuntimeError("indice must be in range [0, size]")


        return ret

    @torch.jit.export
    def lookup_indices(self, tokens: List[str]) -> List[int]:
        r"""
        Args:
            tokens: the tokens used to lookup their corresponding `indices`.

        Returns:
            The 'indices` associated with `tokens`.
        """

        ret = []


        for t in tokens:
            if t in self.vocab_set:
                ret.append(self.vocab_term2num[t])
            else:
                # token get default index
                ret.append(self.default_index)

        return ret

    @torch.jit.export
    def get_stoi(self) -> Dict[str, int]:
        r"""
        Returns:
            Dictionary mapping tokens to indices.
        """
        return self.vocab_term2num

    @torch.jit.export
    def get_itos(self) -> List[str]:
        r"""
        Returns:
            List mapping indices to tokens.
        """
        return self.tokens