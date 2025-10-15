# -*- coding: utf-8 -*-

import os.path
import unicodedata
from functools import reduce
from importlib.resources import files
from typing import Iterable, Iterator, List, Set, Tuple

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


def segment_text(text: str, seg_regex: re.Pattern = SEG_REGEX) -> Iterator[str]:
    """Return an iterator of segments in the text.

    Parameters
    ----------
    text : str
        String of IPA Unicode text to segment.
    seg_regex : re.Pattern, optional
        Compiled regex defining a segment (base + modifiers).
        Default is SEG_REGEX.

    Returns
    -------
    Iterator[str]
        Iterator yielding segments in the input text.
    """
    for m in seg_regex.finditer(text):
        yield m.group(0)


def fts(s: str) -> Set[Tuple[str, str]]:
    """Extract features from a feature specification string.

    Given string `s` with +/-[alphabetical sequence]s, return set of
    features.

    Parameters
    ----------
    s : str
        String with segments of the sort "+son -syl 0cor".

    Returns
    -------
    Set[Tuple[str, str]]
        Set of (value, feature) tuples where value is '+', '-', or '0'.
    """
    return {m.groups() for m in FT_REGEX.finditer(s)}


def pat(p: str) -> List[Set[Tuple[str, str]]]:
    """Parse feature pattern string into list of feature sets.

    Given a string `p` with feature matrices (features grouped with square
    brackets into segments), return a list of sets of (value, feature) tuples.

    Parameters
    ----------
    p : str
        String containing feature matrices as bracketed segments.

    Returns
    -------
    List[Set[Tuple[str, str]]]
        List of sets of (value, feature) tuples, one set per segment.
    """
    pattern = []
    for matrix in [m.group(0) for m in MT_REGEX.finditer(p)]:
        segment = set([m.groups() for m in FT_REGEX.finditer(matrix)])
        pattern.append(segment)
    return pattern


def word2array(ft_names: List[str], word: List[List[Tuple[str, str]]]) -> np.ndarray:
    """Convert word feature representation to NumPy array.

    Given a word consisting of lists of lists/sets of (value, feature) tuples,
    return a NumPy array where each row is a segment and each column is a
    feature.

    Parameters
    ----------
    ft_names : List[str]
        List of feature names (as strings) in order. This argument controls
        what features are included in the array that is output and their
        order vis-a-vis the columns of the array.
    word : List[List[Tuple[str, str]]]
        List of lists of feature tuples (output by FeatureTable.word_fts).

    Returns
    -------
    np.ndarray
        Array in which each row is a segment and each column is a feature.
        Values are 1 for '+', -1 for '-', and 0 for '0'.
    """
    vdict = {'+': 1, '-': -1, '0': 0}

    def seg2col(seg: List[Tuple[str, str]]) -> List[int]:
        seg = dict([(k, v) for (v, k) in seg])
        return [vdict[seg[ft]] for ft in ft_names]
    return np.array([seg2col(s) for s in word], order='F')


class FeatureTable(object):
    """Encapsulate the segment <=> feature mapping.

    This class provides access to the segment-to-feature mapping stored in
    the data file "data/ipa_all.csv" and related functionality for working
    with phonological features.
    """

    def __init__(self, feature_set: str = 'spe+') -> None:
        """Construct a FeatureTable object.

        Parameters
        ----------
        feature_set : str, optional
            The feature set that the FeatureTable will use. Currently,
            there is only one of these ("spe+"). Default is "spe+".
        """
        filename = filenames[feature_set]
        self.segments, self.seg_dict, self.names = self._read_table(filename)
        self.seg_seq = {seg[0]: i for (i, seg) in enumerate(self.segments)}
        self.weights = self._read_weights()
        self.seg_regex = self._build_seg_regex()
        self.longest_seg = max([len(x) for x in self.seg_dict.keys()])
        self.xsampa = xsampa.XSampa()

    @staticmethod
    def normalize(data: str) -> str:
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

    def _read_weights(self, filename: str = os.path.join(
            'data', 'feature_weights.csv')
    ) -> List[float]:
        with files('panphon').joinpath(filename).open() as f:
            df = pd.read_csv(f)
        weights = df.iloc[0].astype(float).tolist()
        return weights

    def _build_seg_regex(self) -> re.Pattern:
        # Build a regex that will match individual segments in a string.
        segs = sorted(self.seg_dict.keys(), key=lambda x: len(x), reverse=True)
        return re.compile(r'(?P<all>{})'.format('|'.join(segs)))

    def fts(self, segment: str) -> List[Tuple[str, str]]:
        """Return features corresponding to segment.

        Returns features corresponding to `segment` as list of (value,
        feature) tuples.

        Parameters
        ----------
        segment : str
            Segment for which features are to be returned as Unicode IPA string.

        Returns
        -------
        List[Tuple[str, str]]
            List of (value, feature) tuples if `segment` is valid,
            otherwise empty list.
        """
        return self.seg_dict.get(segment, [])

    def match(self, ft_mask: Set[Tuple[str, str]], ft_seg: Set[Tuple[str, str]]) -> bool:
        """Check if feature mask matches segment features.

        Answer question "are `ft_mask`'s features a subset of ft_seg?"

        Parameters
        ----------
        ft_mask : Set[Tuple[str, str]]
            Pattern defined as set of (value, feature) tuples.
        ft_seg : Set[Tuple[str, str]]
            Segment defined as a set of (value, feature) tuples.

        Returns
        -------
        bool
            True if and only if all features in `ft_mask` are also in `ft_seg`.
        """
        return set(ft_mask) <= set(ft_seg)

    def fts_match(
            self,
            features: Iterable[tuple[str, str]],
            segment: str | None):
        """Check if features match segment with validity checking.

        Answer question "are `features` a subset of segment's features?"
        This is like `FeatureTable.match` except that it checks whether a
        segment is valid and returns None if it is not.

        Parameters
        ----------
        features : Iterable[Tuple[str, str]]
            Pattern defined as iterable of (value, feature) tuples.
        segment : str or None
            Segment string to check against, or None.

        Returns
        -------
        bool or None
            True if and only if all features in `features` are also in segment's
            features; None if segment is not valid or is None.
        """
        feature_set: set[tuple[str, str]] = set(features)
        seg_fts: list[tuple[str, str]] | None = self.fts(segment)
        if (segment is not None) and seg_fts is not None:
            return feature_set <= set(seg_fts)
        else:
            return None

    def longest_one_seg_prefix(self, word, normalize=True):
        """Return longest single-segment prefix of a word.

        Parameters
        ----------
        word : str
            Input word as Unicode IPA string.
        normalize : bool, optional
            Whether to normalize the word using Unicode NFD. Default is True.

        Returns
        -------
        str
            Longest single-segment prefix of `word` found in database,
            or empty string if no valid prefix exists.
        """
        if normalize:
            word = FeatureTable.normalize(word)

        for i in range(self.longest_seg, 0, -1):
            if word[:i] in self.seg_dict:
                return word[:i]
        return ''

    def validate_word(self, word: str) -> bool:
        """Check if word consists exhaustively of valid IPA segments.

        Parameters
        ----------
        word : str
            Input word as Unicode IPA string.

        Returns
        -------
        bool
            True if `word` can be divided exhaustively into IPA segments
            that exist in the database, False otherwise.
        """
        while word:
            match = self.seg_regex.match(word)
            if match:
                word = word[len(match.group(0)):]
            else:
                return False
        return True

    def segs(self, word: str) -> List[str]:
        """Extract segments from a word.

        Parameters
        ----------
        word : str
            Input word as Unicode IPA string.

        Returns
        -------
        List[str]
            List of strings corresponding to segments found in `word`.
        """
        return [m.group('all') for m in self.seg_regex.finditer(word)]

    def word_fts(self, word: str) -> List[List[Tuple[str, str]]]:
        """Return featural analysis of word.

        Parameters
        ----------
        word : str
            One or more IPA segments.

        Returns
        -------
        List[List[Tuple[str, str]]]
            List of lists of (value, feature) tuples where each inner list
            corresponds to a segment in `word`.
        """
        return list(map(self.fts, self.segs(word)))

    def word_array(self, ft_names: List[str], word: str) -> np.ndarray:
        """Return word as feature array.

        Return `word` as [-1, 0, 1] features in a NumPy array.

        Parameters
        ----------
        ft_names : List[str]
            List of feature names in order.
        word : str
            Word as an IPA string.

        Returns
        -------
        np.ndarray
            Array with segments in rows, features in columns as [-1, 0, 1].
        """
        return word2array(ft_names, self.word_fts(word))

    def seg_known(self, segment: str) -> bool:
        """Check if segment is in the features database.

        Parameters
        ----------
        segment : str
            Consonant or vowel segment.

        Returns
        -------
        bool
            True if `segment` is in the database, False otherwise.
        """
        return segment in self.seg_dict

    def segs_safe(self, word: str) -> List[str]:
        """Extract segments from word with fallback for invalid characters.

        Characters that are not valid segments are included in the list as
        individual characters.

        Parameters
        ----------
        word : str
            Word as an IPA string.

        Returns
        -------
        List[str]
            List of Unicode IPA strings corresponding to segments in `word`.
            Invalid segments are included as individual characters.
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

    def filter_segs(self, segs: List[str]) -> List[str]:
        """Filter list to include only valid segments.

        Given list of strings, return only those which are valid segments.

        Parameters
        ----------
        segs : List[str]
            List of IPA Unicode strings.

        Returns
        -------
        List[str]
            List of IPA Unicode strings identical to `segs` but with
            invalid segments filtered out.
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
