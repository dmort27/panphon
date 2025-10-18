
import copy
import os.path
from importlib.resources import files

import regex as re

import pandas as pd
import yaml

from . import _panphon, xsampa


def flip(s):
    return [(b, a) for (a, b) in s]


def update_ft_set(seg, dia):
    seg = dict(flip(seg))
    seg.update(dia)
    return flip(set(seg.items()))


class PermissiveFeatureTable(_panphon.FeatureTable):
    """Encapsulate the segment <=> feature vector mapping implied by the files
    data/ipa_all.csv and diacritic_definitions.yml. Uses a more permissive
    algorithm for identifying base+diacritic combinations. To avoid a
    combinatorial explosion, it never generates all of the base-diacritic-
    modifier combinations, meaning it cannot easily make statements about the
    whole set of segments."""

    def __init__(
        self,
        feature_set="spe+",
        feature_model="strict",
        ipa_bases=os.path.join("data", "ipa_bases.csv"),
        dias=os.path.join("data", "diacritic_definitions.yml"),
    ):
        """Construct a PermissiveFeatureTable object

        Args:
            feature_set (str): feature system (for API compatibility)
            feature_model (str): feature parsing model (for API compatibility)
            ipa_bases (str): path from panphon root to CSV file definining
                             features of bases (unmodified consonants and
                             vowels)
            dias (str): path from panphon root to YAML file containing rules
                        for diacritics and modifiers
        """
        dias = files("panphon") / dias
        self.bases, self.names = self._read_ipa_bases(ipa_bases)
        self.prefix_dias, self.postfix_dias = self._read_dias(dias)
        self.pre_regex, self.post_regex, self.seg_regex = \
            self._compile_seg_regexes(
                self.bases, self.prefix_dias, self.postfix_dias
            )
        self.xsampa = xsampa.XSampa()
        self.weights = self._read_weights()

    def _read_ipa_bases(
        self, fn: str
    ) -> tuple[dict[str, set[tuple[str, str]]], list[str]]:
        # Open file from package
        path = files("panphon") / fn
        with path.open() as f:
            df = pd.read_csv(f)

        # Compute the
        names = list(df.columns[1:])  # feature names
        bases = {
            row["ipa"]: set(zip(row[1:], names))
            for _, row in df.iterrows()
        }

        return bases, names

    def _read_dias(self, dias):
        prefix, postfix = {}, {}
        with dias.open('r') as f:
            defs = yaml.load(f.read(), Loader=yaml.FullLoader)
            for dia in defs['diacritics']:
                if dia['position'] == 'pre':
                    prefix[dia['marker']] = dia['content']
                else:
                    postfix[dia['marker']] = dia['content']
        return prefix, postfix

    def _compile_seg_regexes(self, bases, prefix, postfix):
        pre_jnd = '|'.join(prefix.keys())
        post_jnd = '|'.join(postfix.keys())
        bases_jnd = '|'.join(bases.keys())
        pre_re = '({})'.format(pre_jnd)
        post_re = '({})'.format(post_jnd)
        seg_re = '(?P<all>(?P<pre>({})*)(?P<base>{})(?P<post>({})*))'.format(
            pre_jnd, bases_jnd, post_jnd
        )
        return re.compile(pre_re), re.compile(post_re), re.compile(seg_re)

    def _build_seg_regex(self):
        return self.seg_regex

    def _read_weights(
            self,
            filename=os.path.join(
                "data", "feature_weights.csv")):
        path = files("panphon").joinpath(filename)
        with path.open() as f:
            df = pd.read_csv(f)

        # Weights are in first row
        weights = df.iloc[0].astype(float).tolist()
        return weights

    def fts(self, segment):
        """Return features corresponding to segment as list of (value,
        feature) tuples

        Args:
            segment (str): segment for which features are to be returned as
                               Unicode string

        Returns:
            list: None if `segment` cannot be parsed; otherwise, a list of the
                  features of `segment` as (value, feature) pairs
        """
        match = self.seg_regex.match(segment)
        if match:
            pre, base, post = \
                match.group('pre'), \
                match.group('base'), \
                match.group('post')
            seg = copy.deepcopy(self.bases[base])
            for m in reversed(pre):
                seg = update_ft_set(seg, self.prefix_dias[m])
            for m in post:
                seg = update_ft_set(seg, self.postfix_dias[m])
            return set(seg)
        else:
            return None

    def fts_match(self, fts_mask, segment):
        """Evaluates whether a set of features 'match' a segment (are a subset
        of that segment's features)

        Args:
            fts_mask (list): list of (value, feature)
                             tuples segment (str):
            IPA string corresponding to segment
                            (consonant or vowel)
        Returns:
            bool: None if `segment` cannot be parsed; True if the feature
                values of `fts_mask` are a subset of those for `segment`
        """
        fts_seg = self.fts(segment)
        if fts_seg:
            fts_mask = set(fts_mask)
            return fts_mask <= fts_seg
        else:
            return None

    def longest_one_seg_prefix(self, word):
        """Return longest IPA Unicode prefix of `word`

        Args:
            word (str): word as IPA string

        Returns:
            str: longest single-segment prefix of `word`
        """
        match = self.seg_regex.match(word)
        if match:
            return match.group(0)
        else:
            return ''

    def seg_known(self, segment):
        """Return True if the segment is valid

        Args:
            segment (str): a string which may or may not be a valid segment

        Returns:
            bool: True if segment can be parsed given the database of bases and
                  diacritics
        """
        if self.seg_regex.match(segment):
            return True
        else:
            return False

    def filter_segs(self, segs):
        """Given list of strings, return only those which are valid segments.

        Args:
            segs (list): list of str values

        Returns:
            list: values in `segs` that are valid segments (according to the
                  definitions of bases and diacritics/modifiers known to the
                  object
        """
        def whole_seg(seg):
            m = self.seg_regex.match(seg)
            if m and m.group(0) == seg:
                return True
            else:
                return False
        return list(filter(whole_seg, segs))

    def segment_word_segments(self, word):
        def n2s(s):
            if s is None:
                return ''
            else:
                return s
        return ((n2s(m.group('pre')),
                 n2s(m.group('base')),
                 n2s(m.group('post')))
                for m in self.seg_regex.finditer(word))

    @property
    def all_segs_matching_fts(self):
        raise AttributeError(
            "'PermissiveFeatureTable' object has no attribute " +
            "'all_segs_matching_fts'"
        )
