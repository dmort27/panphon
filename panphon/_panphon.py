# -*- coding: utf-8 -*-

import os.path
import unicodedata
from functools import reduce
from importlib.resources import files
from typing import Iterable

import numpy as np
import pandas as pd
import regex as re

from . import xsampa

# logging.basicConfig(level=logging.DEBUG)


FT_REGEX = re.compile(r'([-+0])([a-z][A-Za-z]*)', re.U | re.X)
MT_REGEX = re.compile(r'\[[-+0a-zA-Z ,;]*\]')
SEG_REGEX = re.compile(r'[\p{InBasic_Latin}\p{InGreek_and_Coptic}' +
                       r'\p{InIPA_Extensions}ŋœ\u00C0-\u00FF]' +
                       r'[\u0300-\u0360\u0362-\u036F]*' +
                       r'\p{InSpacing_Modifier_Letters}*',
                       re.U | re.X)
filenames = {
    'spe+': os.path.join('data', 'ipa_all.csv'),
    'panphon': os.path.join('data', 'ipa_all.csv'),
}


def segment_text(text, seg_regex=SEG_REGEX):
    """Return an iterator of segments in the text.

    Args:
        text (str): string of IPA Unicode text
        seg_regex (_regex.Pattern): compiled regex defining a segment (base +
                                    modifiers)

    Return:
        generator: segments in the input text
    """
    for m in seg_regex.finditer(text):
        yield m.group(0)


def fts(s: str) -> set[tuple]:
    """
    Given string `s` with +/-[alphabetical sequence]s, return list of
    features.

    Args:
        s (str): string with segments of the sort "+son -syl 0cor"

    Return:
        list: list of (value, feature) tuples
    """
    return {m.groups() for m in FT_REGEX.finditer(s)}


def pat(p):
    """Given a string `p` with feature matrices (features grouped with square
    brackets into segments, return a list of sets of (value, feature) tuples.

    Args:
        p (str): list of feature matrices as strings

    Return:
        list: list of sets of (value, feature) tuples
    """
    pattern = []
    for matrix in [m.group(0) for m in MT_REGEX.finditer(p)]:
        segment = set([m.groups() for m in FT_REGEX.finditer(matrix)])
        pattern.append(segment)
    return pattern


def word2array(ft_names, word):
    """Converts `word` [[(value, feature),...],...] to a NumPy array

    Given a word consisting of lists of lists/sets of (value, feature) tuples,
    return a NumPy array where each row is a segment and each column is a
    feature.

    Args:
        ft_names (list): list of feature names (as strings) in order; this
                         argument controls what features are included in the
                         array that is output and their order vis-a-vis the
                         columns of the array
        word (list): list of lists of feature tuples (output by
                     FeatureTable.word_fts)

    Returns:
        ndarray: array in which each row is a segment and each column
                         is a feature
    """
    vdict = {'+': 1, '-': -1, '0': 0}

    def seg2col(seg):
        seg = dict([(k, v) for (v, k) in seg])
        return [vdict[seg[ft]] for ft in ft_names]
    return np.array([seg2col(s) for s in word], order='F')


class FeatureTable(object):
    """Encapsulate the segment <=> feature mapping in the file
    "data/ipa_all.csv".
    """

    def __init__(self, feature_set='spe+'):
        """Construct a FeatureTable object

        Args:
            feature_set (str): the feature set that the FeatureTable will use;
                               currently, there is only one of these ("spe+")

        """
        filename = filenames[feature_set]
        self.segments, self.seg_dict, self.names = self._read_table(filename)
        self.seg_seq = {seg[0]: i for (i, seg) in enumerate(self.segments)}
        self.weights = self._read_weights()
        self.seg_regex = self._build_seg_regex()
        self.longest_seg = max([len(x) for x in self.seg_dict.keys()])
        self.xsampa = xsampa.XSampa()

    @staticmethod
    def normalize(data):
        return unicodedata.normalize('NFD', data)

    def _read_table(self, filename: str) -> tuple[
        Iterable[tuple[str, list[tuple[str, str]]]],
        dict[str, list[tuple[str, str]]],
        list[str]
    ]:
        """
        Read the data from a CSV into:
        - a zip iterator of (ipa, features) tuples,
        - a dict mapping IPA strings to their features,
        - and a list of feature names.
        """
        with files('panphon').joinpath(filename).open('rb') as f:
            df = pd.read_csv(f)

        header: list[str] = list(df.columns)
        names: list[str] = header[1:]

        ft_dicts: list[dict[str, str]] = \
            df[names].to_dict(orient='records')  # type: ignore

        specs: list[list[tuple[str, str]]] = [
            [(v, n) for n, v in ft_dict.items()]
            for ft_dict in ft_dicts
        ]

        ipa: Iterable[str] = df['ipa']
        segments: Iterable[tuple[str, list[tuple[str, str]]]] = \
            zip(ipa, specs)
        seg_dict: dict[str, list[tuple[str, str]]] = dict(segments)

        return list(zip(ipa, specs))[1:], seg_dict, names

    def _read_weights(self, filename=os.path.join(
            'data', 'feature_weights.csv')
    ):
        with files('panphon').joinpath(filename).open() as f:
            df = pd.read_csv(f)
        weights = df.iloc[0].astype(float).tolist()
        return weights

    def _build_seg_regex(self):
        # Build a regex that will match individual segments in a string.
        segs = sorted(self.seg_dict.keys(), key=lambda x: len(x), reverse=True)
        return re.compile(r'(?P<all>{})'.format('|'.join(segs)))

    def fts(self, segment: str) -> list[tuple[str, str]]:
        """Returns features corresponding to `segment` as list of (value,
        feature) tuples.

        Args:
           segment (str): segment for which features are to be returned as
                              Unicode IPA string.

        Returns:
            set: set of (value, feature) tuples, if `segment` is valid;
            otherwise,
                 None
        """
        return self.seg_dict.get(segment, [])

    def match(self, ft_mask, ft_seg):
        """Answer question "are `ft_mask`'s features a subset of ft_seg?"

        Args:
            ft_mask (set): pattern defined as set of (value, feature) tuples
            ft_seg (set): segment defined as a set of (value, feature) tuples

        Returns:
            bool: True iff all features in `ft_mask` are also in `ft_seg`
        """
        return set(ft_mask) <= set(ft_seg)

    def fts_match(
            self,
            features: Iterable[tuple[str, str]],
            segment: str | None):
        """Answer question "are `ft_mask`'s features a subset of ft_seg?"

        This is like `FeatureTable.match` except that it checks whether a
        segment is valid and returns None if it is not.

        Args:
            features (set): pattern defined as set of (value, feature) tuples
            segment (set): segment defined as a set of (value, feature) tuples

        Returns:
            bool: True iff all features in `ft_mask` are also in `ft_seg`; None
                  if segment is not valid
        """
        feature_set: set[tuple[str, str]] = set(features)
        seg_fts: list[tuple[str, str]] | None = self.fts(segment)
        if (segment is not None) and seg_fts is not None:
            return feature_set <= set(seg_fts)
        else:
            return None

    def longest_one_seg_prefix(self, word, normalize=True):
        """Return longest Unicode IPA prefix of a word

        Args:
            word (str): input word as Unicode IPA string

        Returns:
            str: longest single-segment prefix of `word` in database
        """
        if normalize:
            word = FeatureTable.normalize(word)

        for i in range(self.longest_seg, 0, -1):
            if word[:i] in self.seg_dict:
                return word[:i]
        return ''

    def validate_word(self, word):
        """Returns True if `word` consists exhaustively of valid IPA segments

        Args:
            word (str): input word as Unicode IPA string

        Returns:
            bool: True if `word` can be divided exhaustively into IPA segments
                  that exist in the database

        """
        while word:
            match = self.seg_regex.match(word)
            if match:
                word = word[len(match.group(0)):]
            else:
                return False
        return True

    def segs(self, word):
        """Returns a list of segments from a word

        Args:
            word (str): input word as Unicode IPA string

        Returns:
            list: list of strings corresponding to segments found in `word`
        """
        return [m.group('all') for m in self.seg_regex.finditer(word)]

    def word_fts(self, word):
        """Return featural analysis of `word`

        Args:
            word (str):  one or more IPA segments

        Returns:
            list: list of lists (value, feature) tuples where each inner list
                  corresponds to a segment in `word`
        """
        return list(map(self.fts, self.segs(word)))

    def word_array(self, ft_names, word):
        """Return `word` as [-1, 0, 1] features in a NumPy array

        Args:
            ft_names (list): list of feature names in order
            word (str): word as an IPA string

        Returns:
            ndarray: segments in rows, features in columns as [-1, 0 , 1]
        """
        return word2array(ft_names, self.word_fts(word))

    def seg_known(self, segment):
        """Return True if `segment` is in segment <=> features database

        Args:
            segment (str): consonant or vowel

        Returns:
            bool: True, if `segment` is in the database
        """
        return segment in self.seg_dict

    def segs_safe(self, word):
        """Return a list of segments (as strings) from a word

        Characters that are not valid segments are included in the list as
        individual characters.

        Args:
            word (str): word as an IPA string

        Returns:
            list: list of Unicode IPA strings corresponding to segments in
                  `word`
        """
        segs = []
        while word:
            m = self.seg_regex.match(word)
            if m:
                segs.append(m.group(1))
                word = word[len(m.group(1)):]
            else:
                segs.append(word[0])
                word = word[1:]
        return segs

    def filter_segs(self, segs: list[str]) -> list[str]:
        """Given list of strings, return only those which are valid segments

        Args:
            segs (list): list of IPA Unicode strings

        Return:
            list: list of IPA Unicode strings identical to `segs` but with
                  invalid segments filtered out
        """
        return list(filter(self.seg_known, segs))

    def filter_string(self, word):
        """Return a string like the input but containing only legal IPA 
        segments

        Args:
            word (str): input string to be filtered

        Returns:
            str: string identical to `word` but with invalid IPA segments
                     absent

        """
        segs = [m.group(0) for m in self.seg_regex.finditer(word)]
        return ''.join(segs)

    def fts_intersection(self, segs: list[str]) -> set[tuple[str, str]]:
        """Return the features shared by `segs`

        Args:
            segs (list): list of Unicode IPA segments

        Returns:
            set: set of (value, feature) tuples shared by the valid segments in
                 `segs`
        """
        fts_vecs: list[tuple[str, str]] | None = [
            self.fts(s) for s in self.filter_segs(segs)]
        return reduce(lambda a, b: set(a) & set(b), fts_vecs)

    def fts_match_any(self, fts, inv):
        """Return `True` if any segment in `inv` matches the features in `fts`

        Args:
            fts (list): a collection of (value, feature) tuples
            inv (list): a collection of IPA segments represented as Unicode
                        strings

        Returns:
            bool: `True` if any segment in `inv` matches the features in `fts`
        """
        return any([self.fts_match(fts, s) for s in inv])

    def fts_match_all(self, fts, inv):
        """Return `True` if all segments in `inv` matches the features in fts

        Args:
            fts (list): a collection of (value, feature) tuples
            inv (list): a collection of IPA segments represented as Unicode
                        strings

        Returns:
            bool: `True` if all segments in `inv` matches the features in `fts`
        """
        return all([self.fts_match(fts, s) for s in inv])

    def fts_contrast2(self, fs, ft_name, inv):
        """Return `True` if there is a segment in `inv` that contrasts in feature
        `ft_name`.

        Args:
            fs (list): feature specifications used to filter `inv`.
            ft_name (str): name of the feature where contrast must be present.
            inv (list): collection of segments represented as Unicode segments.

        Returns:
            bool: `True` if two segments in `inv` are identical in features except
                  for feature `ft_name`
        """
        inv_fts = [
            set(self.fts(x))
            for x in inv
            if set(fs) <= set(self.fts(x))
        ]
        for a in inv_fts:
            for b in inv_fts:
                if a != b:
                    diff = a ^ b
                    if len(diff) == 2:
                        if all([nm == ft_name for (_, nm) in diff]):
                            return True
        return False

    def fts_count(self, fts, inv):
        """Return the count of segments in an inventory matching a given
        feature mask.

        Args:
            fts (set): feature mask given as a set of (value, feature) tuples
            inv (set): inventory of segments (as Unicode IPA strings)

        Returns:
            int: number of segments in `inv` that match feature mask `fts`
        """
        return len(list(filter(lambda s: self.fts_match(fts, s), inv)))

    def match_pattern(self, pat, word):
        """Implements fixed-width pattern matching.

        Matches just in case pattern is the same length (in segments) as the
        word and each of the segments in the pattern is a featural subset of the
        corresponding segment in the word. Matches return the corresponding list
        of feature sets; failed matches return None.

        Args:
           pat (list): pattern consisting of a sequence of sets of (value,
                       feature) tuples
           word (str): a Unicode IPA string consisting of zero or more
                          segments

        Returns:
            list: corresponding list of feature sets or, if there is no match,
                  None
        """
        segs = self.word_fts(word)
        if len(pat) != len(segs):
            return None
        else:
            if all([set(p) <= set(s) for (p, s) in zip(pat, segs)]):
                return segs

    def match_pattern_seq(self, pat, const):
        """Implements limited pattern matching. Matches just in case pattern is
        the same length (in segments) as the constituent and each of the
        segments in the pattern is a featural subset of the corresponding
        segment in the word.

        Args:
            pat (list): pattern consisting of a list of sets of (value, feature)
                        tuples.
            const (list): a sequence of Unicode IPA strings consisting of zero
                          or more segments.

        Returns:
            bool: `True` if `const` matches `pat`
        """
        segs = [self.fts(s) for s in const]
        if len(pat) != len(segs):
            return False
        else:
            return all([set(p) <= set(s) for (p, s) in zip(pat, segs)])

    def all_segs_matching_fts(self, fts):
        """Return segments matching a feature mask, both as (value, feature)
        tuples (sorted in reverse order by length).

        Args:
            fts (list): feature mask as (value, feature) tuples.

        Returns:
            list: segments matching `fts`, sorted in reverse order by length
        """
        matching_segs = []
        for seg, pairs in self.segments:
            if set(fts) <= set(pairs):
                matching_segs.append(seg)
        return sorted(matching_segs, key=lambda x: len(x), reverse=True)

    def compile_regex_from_str(self, ft_str):
        """Given a string describing features masks for a sequence of segments,
        return a regex matching the corresponding strings.

        Args:
            ft_str (str): feature masks, each enclosed in square brackets, in
            which the features are delimited by any standard delimiter.

        Returns:
           Pattern: regular expression pattern equivalent to `ft_str`
        """

        sequence = []
        for m in re.finditer(r'\[([^]]+)\]', ft_str):
            ft_mask = fts(m.group(1))
            segs = self.all_segs_matching_fts(ft_mask)
            sub_pat = '({})'.format('|'.join(segs))
            sequence.append(sub_pat)
        pattern = ''.join(sequence)
        regex = re.compile(pattern)
        return regex

    def segment_to_vector(self, seg):
        """Given a Unicode IPA segment, return a list of feature specificiations
        in cannonical order.

        Args:
            seg (str): IPA consonant or vowel

        Returns:
            list: feature specifications ('+'/'-'/'0') in the order from
            `FeatureTable.names`
        """
        ft_dict = {ft: val for (val, ft) in self.fts(seg)}
        return [ft_dict[name] for name in self.names]

    def tensor_to_numeric(self, t):
        return list(map(lambda a:
                    list(map(lambda b: {'+': 1, '-': -1, '0': 0}[b], a)), t))

    def word_to_vector_list(self, word, numeric=False, xsampa=False):
        """Return a list of feature vectors, given a Unicode IPA word.

        Args:
            word (str): string in IPA
            numeric (bool): if True, return features as numeric values instead
                            of strings

        Returns:
            list: a list of lists of '+'/'-'/'0' or 1/-1/0
        """
        if xsampa:
            word = self.xsampa.convert(word)
        tensor = list(map(self.segment_to_vector, self.segs(word)))
        if numeric:
            return self.tensor_to_numeric(tensor)
        else:
            return tensor
