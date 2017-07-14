# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path

import numpy
import pkg_resources

import regex as re
import unicodecsv as csv

from . import xsampa
from .segment import Segment

feature_sets = {
    'spe+': (os.path.join('data', 'ipa_all.csv'),
             os.path.join('data', 'feature_weights.csv'))
}


class FeatureTable(object):
    def __init__(self, feature_set='spe+'):
        bases_fn, weights_fn = feature_sets[feature_set]
        self.weights = self._read_weights(weights_fn)
        self.segments, self.seg_dict, self.names = self._read_bases(bases_fn, self.weights)
        self.seg_regex = self._build_seg_regex()
        self.longest_seg = max([len(x) for x in self.seg_dict.keys()])
        self.xsampa = xsampa.XSampa()

    def _read_bases(self, fn, weights):
        fn = pkg_resources.resource_filename(__name__, fn)
        segments = []
        with open(fn, 'rb') as f:
            reader = csv.reader(f, encoding='utf-8')
            header = next(reader)
            names = header[1:]
            for row in reader:
                ipa = row[0]
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

    def fts(self, ipa):
        if ipa in self.seg_dict:
            return self.seg_dict[ipa]
        else:
            return None

    def longest_one_seg_prefix(self, word):
        """Return longest Unicode IPA prefix of a word

        Args:
            word (unicode): input word as Unicode IPA string

        Returns:
            unicode: longest single-segment prefix of `word` in database
        """
        for i in range(self.longest_seg, 0, -1):
            if word[:i] in self.seg_dict:
                return word[:i]
        return ''

    def ipa_segs(self, word):
        """Returns a list of segments from a word

        Args:
            word (unicode): input word as Unicode IPA string

        Returns:
            list: list of strings corresponding to segments found in `word`
        """
        return [m.group('all') for m in self.seg_regex.finditer(word)]

    def validate_word(self, word):
        """Returns True if `word` consists exhaustively of valid IPA segments

        Args:
            word (unicode): input word as Unicode IPA string

        Returns:
            bool: True if `word` can be divided exhaustively into IPA segments
                  that exist in the database

        """
        return word == ''.join(self.ipa_segs(word))

    def word_fts(self, word):
        """Return a list of Segment objects corresponding to the segments in
           word.

        Args:
            word (unicode): word consisting of IPA segments

        Returns:
            list: list of Segment objects corresponding to word
        """
        return [self.fts(ipa) for ipa in self.ipa_segs(word)]

    def word_array(self, ft_names, word):
        """Return a nparray of features namd in ft_name for the segments in word

        Args:
            ft_names (list): strings naming subset of features in self.names
            word (unicode): word to be analyzed

        Returns:
            ndarray: segments in rows, features in columns as [-1, 0, 1]
        """
        return numpy.array([s.numeric(ft_names) for s in self.word_fts(word)])

    def seg_known(self, segment):
        """Return True if `segment` is in segment <=> features database

        Args:
            segment (unicode): consonant or vowel

        Returns:
            bool: True, if `segment` is in the database
        """
        return segment in self.seg_dict

    def segs_safe(self, word):
        """Return a list of segments (as strings) from a word

        Characters that are not valid segments are included in the list as
        individual characters.

        Args:
            word (unicode): word as an IPA string

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

    def filter_segs(self, segs):
        """Given list of strings, return only those which are valid segments

        Args:
            segs (list): list of IPA Unicode strings

        Return:
            list: list of IPA Unicode strings identical to `segs` but with
                  invalid segments filtered out
        """
        return list(filter(self.seg_known, segs))

    def filter_string(self, word):
        """Return a string like the input but containing only legal IPA segments

        Args:
            word (unicode): input string to be filtered

        Returns:
            unicode: string identical to `word` but with invalid IPA segments
                     absent

        """
        segs = [m.group(0) for m in self.seg_regex.finditer(word)]
        return ''.join(segs)

    def fts_intersection(self, segs):
        """Return a Segment object containing the features shared by all segments

        Args:
            segs (list): IPA segments

        Returns:
            Segment: the features shared by all segments in segs
        """
        return reduce(lambda a, b: a & b,
                      [self.fts(s) for s in self.filter_segs(segs)])

    def fts_match_all(self, fts, inv):
        """Return `True` if all segments in `inv` matches the features in fts

        Args:
            fts (dict): a dictionary of features
            inv (list): a collection of IPA segments represented as Unicode
                        strings

        Returns:
            bool: `True` if all segments in `inv` matches the features in `fts`
        """
        return all([self.fts(s) >= fts for s in inv])

    def fts_match_any(self, fts, inv):
        """Return `True` if any segments in `inv` matches the features in fts

        Args:
            fts (dict): a dictionary of features
            inv (list): a collection of IPA segments represented as Unicode
                        strings

        Returns:
            bool: `True` if any segments in `inv` matches the features in `fts`
        """
        return any([self.fts(s) >= fts for s in inv])

    def fts_contrast2(self, fs, ft_name, inv):
        """Return `True` if there is a segment in `inv` that contrasts in feature
        `ft_name`.

        Args:
            fs (dict): feature specifications used to filter `inv`.
            ft_name (str): name of the feature where contrast must be present.
            inv (list): collection of segments represented as Unicode strings.

        Returns:
            bool: `True` if two segments in `inv` are identical in features except
                  for feature `ft_name`
        """
        inv_segs = [self.fts(x) for x in inv if self.fts(x) >= fs]
        for a in inv_segs:
            for b in inv_segs:
                if a != b:
                    diff = a ^ b
                    if len(diff) == 2:
                        if all([nm == ft_name for (_, nm) in diff]):
                            return True
        return False
