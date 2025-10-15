import unicodedata
from typing import List
from importlib import resources
import pandas as pd
import marisa_trie


class Tokenizer:
    """Tokenize strings into sequences of phonemes."""
    def __init__(self, ipa_file='ipa_all.csv'):
        with resources.files("panphon.data").joinpath(ipa_file).open("r") as f:
            self.df = pd.read_csv(f)
        self.phonemes = list(self.df['ipa'])
        self.phonemes_bytes = [(p.encode('utf-8'), ) for p in self.df['ipa']]
        self.pairs = list(zip(self.phonemes, self.phonemes_bytes))
        # print(f'self.pairs = {self.pairs[:10]} ... {self.pairs[-10:]}')
        self.trie = marisa_trie.RecordTrie('@s', self.pairs)

    def prefixes(self, s: str) -> List[str]:
        """Return all prefixes of s that are in the trie."""
        return self.trie.prefixes(s)

    def longest_prefix(self, s: str) -> str:
        """Return the longest prefix of s that is in the trie."""
        prefixes = self.prefixes(s)
        if not prefixes:
            return ''
        else:
            return sorted(prefixes, key=len)[-1]

    def tokenize(self, ipa: str) -> List[str]:
        """Convert IPA string into a sequence of phoneme tokens

        Args:
            ipa (unicode): An IPA string as unicode

        Returns:
            list: a list of strings corresponding to phonemes

            Non-IPA segments are skipped.
        """
        tokens = []
        ipa = unicodedata.normalize('NFD', ipa)
        while ipa:
            token = self.longest_prefix(ipa)
            if token:
                tokens.append(token)
                ipa = ipa[len(token):]
            else:
                ipa = ipa[1:]
        return tokens