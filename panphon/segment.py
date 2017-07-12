# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import regex as re


class Segment(object):
    def __init__(self, names, features={}, ftstr=''):
        self.n2s = {-1: '-', 0: '0', 1: '+'}
        self.s2n = {k: v for (v, k) in self.n2s.items()}
        self.names = names
        self.data = {}
        for name in names:
            if name in features:
                self.data[name] = features[name]
            else:
                self.data[name] = 0
        for m in re.finditer('([+0-])(\w+)', ftstr):
            v, k = m.groups()
            self.data[k] = self.s2n[v]

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        if key in names:
            self.data[key] = value
        else:
            raise KeyError('Unknown feature name.')

    def __repr__(self):
        pairs = [(self.n2s[self.data[k]], k) for k in self.names]
        fts = ', '.join(['{}{}'.format(*pair) for pair in pairs])
        return '[{}]'.format(fts)

    def __iter__(self):
        return iter(self.names)

    def iteritems(self):
        return iter([(k, self.data[k]) for k in self.names])

    def update(self, segment):
        self.data.update(segment)

    def match(self, features):
        return all([self.data[k] == v for (k, v) in features.items()])

    def __ge__(self, other):
        return self.match(other)

    def numeric(self):
        return [self.data[k] for k in self.names]

    def string(self):
        return map(lambda x: self.tr[x], self.numeric())
