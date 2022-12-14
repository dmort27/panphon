# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
import unicodedata
import collections

import numpy
import pkg_resources

import regex as re
import unicodecsv as csv

from . import xsampa
from .segment import Segment
from functools import reduce

feature_sets = {
    'spe+': (os.path.join('data', 'ipa_all.csv'),
             os.path.join('data', 'feature_weights.csv'))
}


class FeatureTable(object):
    TRIE_LEAF_MARKER = None

    def __init__(self, feature_set='spe+'):
        bases_fn, weights_fn = feature_sets[feature_set]
        self.weights = self._read_weights(weights_fn)
        self.segments, self.seg_dict, self.names = self._read_bases(bases_fn, self.weights)
        self.seg_regex = self._build_seg_regex()
        self.seg_trie = self._build_seg_trie()
        self.longest_seg = max([len(x) for x in self.seg_dict.keys()])
        self.xsampa = xsampa.XSampa()

    @staticmethod
    def normalize(data):
        return unicodedata.normalize('NFD', data)

    def _read_bases(self, fn, weights):
        fn = pkg_resources.resource_filename(__name__, fn)
        segments = []
        with open(fn, 'rb') as f:
            reader = csv.reader(f, encoding='utf-8')
            header = next(reader)
            names = header[1:]
            for row in reader:
                ipa = FeatureTable.normalize(row[0])
                vals = [{'-': -1, '0': 0, '+': 1}[x] for x in row[1:]]
                vec = Segment(names,
                              {n: v for (n, v) in zip(names, vals)},
                              weights=weights)
                segments.append((ipa, vec))
        seg_dict = dict(segments)
        return segments, seg_dict, names

    def _read_weights(self, weights_fn):
        weights_fn = pkg_resources.resource_filename(__name__, weights_fn)
        with open(weights_fn, 'rb') as f:
            reader = csv.reader(f, encoding='utf-8')
            next(reader)
            weights = [float(x) for x in next(reader)]
        return weights

    def _build_seg_regex(self):
        segs = sorted(self.seg_dict.keys(), key=lambda x: len(x), reverse=True)
        return re.compile(r'(?P<all>{})'.format('|'.join(segs)))

    def _build_seg_trie(self):
        trie = {}
        for seg in self.seg_dict.keys():
            node = trie
            for char in seg:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node[self.TRIE_LEAF_MARKER] = None
        return trie

    def fts(self, ipa, normalize=True):
        if normalize:
            ipa = FeatureTable.normalize(ipa)
        if ipa in self.seg_dict:
            return self.seg_dict[ipa]
        else:
            return None

    def longest_one_seg_prefix(self, word, normalize=True):
        """Return longest Unicode IPA prefix of a word

        Args:
            word (unicode): input word as Unicode IPA string
            normalize (bool): whether the word should be pre-normalized

        Returns:
            unicode: longest single-segment prefix of `word` in database
        """
        if normalize:
            word = FeatureTable.normalize(word)
        last_found_length = 0
        node = self.seg_trie
        for pos in range(len(word) + 1):
            if pos == len(word) or word[pos] not in node:
                return word[:last_found_length]
            node = node[word[pos]]
            if self.TRIE_LEAF_MARKER in node:
                last_found_length = pos + 1

    def ipa_segs(self, word, normalize=True):
        """Returns a list of segments from a word

        Args:
            word (unicode): input word as Unicode IPA string
            normalize (bool): whether to pre-normalize the word

        Returns:
            list: list of strings corresponding to segments found in `word`
        """
        if normalize:
            word = FeatureTable.normalize(word)
        return self._segs(word, include_invalid=False, normalize=normalize)

    def validate_word(self, word, normalize=True):
        """Returns True if `word` consists exhaustively of valid IPA segments

        Args:
            word (unicode): input word as Unicode IPA string
            normalize (bool): whether to pre-normalize the word

        Returns:
            bool: True if `word` can be divided exhaustively into IPA segments
                  that exist in the database

        """
        return not self._segs(word, include_valid=False, include_invalid=True, normalize=normalize)

    def word_fts(self, word, normalize=True):
        """Return a list of Segment objects corresponding to the segments in
           word.

        Args:
            word (unicode): word consisting of IPA segments
            normalize (bool): whether to pre-normalize the word

        Returns:
            list: list of Segment objects corresponding to word
        """
        return [self.fts(ipa, False) for ipa in self.ipa_segs(word, normalize)]

    def word_array(self, ft_names, word, normalize=True):
        """Return a nparray of features namd in ft_name for the segments in word

        Args:
            ft_names (list): strings naming subset of features in self.names
            word (unicode): word to be analyzed
            normalize (bool): whether to pre-normalize the word

        Returns:
            ndarray: segments in rows, features in columns as [-1, 0, 1]
        """
        return numpy.array([s.numeric(ft_names) for s in self.word_fts(word, normalize)])

    def bag_of_features(self, word, normalize=True):
        """Return a vector in which each dimension is the number of times a feature-value pair occurs in the word
        
        Args:
            word (unicode): word consisting of IPA segments
            normalize (bool): whether to pre-normalize the word

        Returns:
            array: array of integers corresponding to a bag of feature-value pair counts
        """
        word_features = self.word_fts(word, normalize)
        features = [v + f for f in self.names for v in ['+', '0', '-']]
        bag = collections.OrderedDict()
        for f in features:
            bag[f] = 0
        vdict = {-1: '-', 0: '0', 1: '+'}
        for w in word_features:
            for (f, v) in w.items():
                bag[vdict[v] + f] += 1
        return numpy.array(list(bag.values()))

    def seg_known(self, segment, normalize=True):
        """Return True if `segment` is in segment <=> features database

        Args:
            segment (unicode): consonant or vowel
            normalize (bool): whether to pre-normalize the segment

        Returns:
            bool: True, if `segment` is in the database
        """
        if normalize:
            segment = FeatureTable.normalize(segment)
        return segment in self.seg_dict

    def segs_safe(self, word, normalize=True):
        """Return a list of segments (as strings) from a word

        Characters that are not valid segments are included in the list as
        individual characters.

        Args:
            word (unicode): word as an IPA string
            normalize (bool): whether to pre-normalize the word

        Returns:
            list: list of Unicode IPA strings corresponding to segments in
                  `word`
        """
        if normalize:
            word = FeatureTable.normalize(word)
        return self._segs(word, include_invalid=True, normalize=normalize)

    def _segs(self, word, *, include_valid=True, include_invalid, normalize):
        if normalize:
            word = FeatureTable.normalize(word)
        segs = []
        while word:
            m = self.longest_one_seg_prefix(word, False)
            if m:
                if include_valid:
                    segs.append(m)
                word = word[len(m):]
            else:
                if include_invalid:
                    segs.append(word[0])
                word = word[1:]
        return segs

    def filter_segs(self, segs, normalize=True):
        """Given list of strings, return only those which are valid segments

        Args:
            segs (list): list of IPA Unicode strings
            normalize (bool): whether to pre-normalize the segments

        Return:
            list: list of IPA Unicode strings identical to `segs` but with
                  invalid segments filtered out
        """
        return list(filter(lambda seg: self.seg_known(seg, normalize), segs))

    def filter_string(self, word, normalize=True):
        """Return a string like the input but containing only legal IPA segments

        Args:
            word (unicode): input string to be filtered
            normalize (bool): whether to pre-normalize the word (and return a normalized string)

        Returns:
            unicode: string identical to `word` but with invalid IPA segments
                     absent

        """
        return ''.join(self.ipa_segs(word, normalize))

    def fts_intersection(self, segs, normalize=True):
        """Return a Segment object containing the features shared by all segments

        Args:
            segs (list): IPA segments
            normalize (bool): whether to pre-normalize the segments

        Returns:
            Segment: the features shared by all segments in segs
        """
        return reduce(lambda a, b: a & b,
                      [self.fts(s, normalize) for s in self.filter_segs(segs, normalize)])

    def fts_match_all(self, fts, inv, normalize=True):
        """Return `True` if all segments in `inv` matches the features in fts

        Args:
            fts (dict): a dictionary of features
            inv (list): a collection of IPA segments represented as Unicode
                        strings
            normalize (bool): whether to pre-normalize the segments

        Returns:
            bool: `True` if all segments in `inv` matches the features in `fts`
        """
        return all([self.fts(s, normalize) >= fts for s in inv])

    def fts_match_any(self, fts, inv, normalize=True):
        """Return `True` if any segments in `inv` matches the features in fts

        Args:
            fts (dict): a dictionary of features
            inv (list): a collection of IPA segments represented as Unicode
                        strings
            normalize (bool): whether to pre-normalize the segments

        Returns:
            bool: `True` if any segments in `inv` matches the features in `fts`
        """
        return any([self.fts(s, normalize) >= fts for s in inv])

    def fts_contrast(self, fs, ft_name, inv, normalize=True):
        """Return `True` if there is a segment in `inv` that contrasts in feature
        `ft_name`.

        Args:
            fs (dict): feature specifications used to filter `inv`.
            ft_name (str): name of the feature where contrast must be present.
            inv (list): collection of segments represented as Unicode strings.
            normalize (bool): whether to pre-normalize the segments

        Returns:
            bool: `True` if two segments in `inv` are identical in features except
                  for feature `ft_name`
        """
        inv_segs = filter(lambda x: x >= fs, map(lambda seg: self.fts(seg, normalize), inv))
        for a in inv_segs:
            for b in inv_segs:
                if a != b:
                    if a.differing_specs(b) == [ft_name]:
                        return True
        return False

    def fts_count(self, fts, inv, normalize=True):
        """Return the count of segments in an inventory matching a given
        feature mask.

        Args:
            fts (dict): feature mask given as a set of (value, feature) tuples
            inv (list): inventory of segments (as Unicode IPA strings)
            normalize (bool): whether to pre-normalize the segments

        Returns:
            int: number of segments in `inv` that match feature mask `fts`
        """
        return len(list(filter(lambda s: self.fts(s, normalize) >= fts, inv)))

    def match_pattern(self, pat, word, normalize=True):
        """Implements fixed-width pattern matching.

        Matches just in case pattern is the same length (in segments) as the
        word and each of the segments in the pattern is a featural subset of the
        corresponding segment in the word. Matches return the corresponding list
        of feature sets; failed matches return None.

        Args:
           pat (list): pattern consisting of a sequence of feature dicts
           word (unicode): a Unicode IPA string consisting of zero or more
                           segments
           normalize (bool): whether to pre-normalize the word

        Returns:
            list: corresponding list of feature dicts or, if there is no match,
                  None
        """
        segs = self.word_fts(word, normalize)
        if len(pat) != len(segs):
            return None
        else:
            if all([s >= p for (s, p) in zip(segs, pat)]):
                return segs

    def match_pattern_seq(self, pat, const, normalize=True):
        """Implements limited pattern matching. Matches just in case pattern is
        the same length (in segments) as the constituent and each of the
        segments in the pattern is a featural subset of the corresponding
        segment in the word.

        Args:
            pat (list): pattern consisting of a list of feature dicts, e.g.
                        [{'voi': 1}]
            const (list): a sequence of Unicode IPA strings consisting of zero
                          or more segments.
            normalize (bool): whether to pre-normalize the segments

        Returns:
            bool: `True` if `const` matches `pat`
        """
        segs = [self.fts(s, normalize) for s in const]
        if len(pat) != len(segs):
            return False
        else:
            return all([s >= p for (s, p) in zip(segs, pat)])

    def all_segs_matching_fts(self, ft_mask):
        """Return segments matching a feature mask, a dict of features

        Args:
            ft_mask (list): feature mask dict, e.g. {'voi': -1, 'cont': 1}.

        Returns:
            list: segments matching `ft_mask`, sorted in reverse order by length
        """
        matching_segs = [ipa for (ipa, fts) in self.segments if fts >= ft_mask]
        return sorted(matching_segs, key=lambda x: len(x), reverse=True)

    def compile_regex_from_str(self, pat):
        """Given a string describing features masks for a sequence of segments,
        return a compiled regex matching the corresponding strings.

        Args:
            pat (str): feature masks, each enclosed in square brackets, in
            which the features are delimited by any standard delimiter.

        Returns:
           Pattern: regular expression pattern equivalent to `pat`
        """
        s2n = {'-': -1, '0': 0, '+': 1}
        seg_res = []
        for mat in re.findall(r'\[[^]]+\]+', pat):
            ft_mask = {k: s2n[v] for (v, k) in re.findall(r'([+-])(\w+)', mat)}
            segs = self.all_segs_matching_fts(ft_mask)
            seg_res.append('({})'.format('|'.join(segs)))
        regexp = ''.join(seg_res)
        return re.compile(regexp)

    def segment_to_vector(self, seg, normalize=True):
        """Given a Unicode IPA segment, return a list of feature specificiations
        in canonical order.

        Args:
            seg (unicode): IPA consonant or vowel
            normalize: whether to pre-normalize the segment

        Returns:
            list: feature specifications ('+'/'-'/'0') in the order from
            `FeatureTable.names`
        """
        return self.fts(seg, normalize).strings()

    def word_to_vector_list(self, word, numeric=False, xsampa=False, normalize=True):
        """Return a list of feature vectors, given a Unicode IPA word.

        Args:
            word (unicode): string in IPA (or X-SAMPA, provided `xsampa` is True)
            numeric (bool): if True, return features as numeric values instead
                            of strings
            xsampa (bool): whether the word is in X-SAMPA instead of IPA
            normalize: whether to pre-normalize the word (applies to IPA only)

        Returns:
            list: a list of lists of '+'/'-'/'0' or 1/-1/0
        """
        if xsampa:
            word = self.xsampa.convert(word)
        segs = self.word_fts(word, normalize or xsampa)
        if numeric:
            tensor = [x.numeric() for x in segs]
        else:
            tensor = [x.strings() for x in segs]
        return tensor
