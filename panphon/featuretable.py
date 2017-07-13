# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path

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
