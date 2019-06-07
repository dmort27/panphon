# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import regex as re


class Segment(object):
    """Models a phonological segment as a vector of features."""
    def __init__(self, names, features={}, ftstr='', weights=None):
        """Construct a `Segment` object

        Args:
            names (list): ordered list of feature names
            features (dict): name-value pairs for specified features
            ftstr (unicode): a string, each /(+|0|-)\w+/ sequence of which is
                             interpreted as a feature specification
            weights (float): order list of feature weights/saliences
            """
        self.n2s = {-1: '-', 0: '0', 1: '+'}
        self.s2n = {k: v for (v, k) in self.n2s.items()}
        self.names = names
        """Set a feature specification"""
        self.data = {}
        for name in names:
            if name in features:
                self.data[name] = features[name]
            else:
                self.data[name] = 0
        for m in re.finditer(r'(\+|0|-)(\w+)', ftstr):
            v, k = m.groups()
            self.data[k] = self.s2n[v]
        if weights:
            self.weights = weights
        else:
            self.weights = [1 for _ in names]

    def __getitem__(self, key):
        """Get a feature specification"""
        return self.data[key]

    def __setitem__(self, key, value):
        """Set a feature specification"""
        if key in self.names:
            self.data[key] = value
        else:
            raise KeyError('Unknown feature name.')

    def __repr__(self):
        """Return a string representation of a feature vector"""
        pairs = [(self.n2s[self.data[k]], k) for k in self.names]
        fts = ', '.join(['{}{}'.format(*pair) for pair in pairs])
        return '<Segment [{}]>'.format(fts)

    def __iter__(self):
        """Return an iterator over the feature names"""
        return iter(self.names)

    def items(self):
        """Return a list of the features as (name, value) pairs"""
        return [(k, self.data[k]) for k in self.names]

    def iteritems(self):
        """Return an iterator over the features as (name, value) pairs"""
        return ((k, self.data[k]) for k in self.names)

    def update(self, features):
        """Update the objects features to match `features`.

        Args:
            features (dict): dictionary containing the new feature values
        """
        self.data.update(features)

    def match(self, ft_mask):
        """Determine whether `self`'s features are a superset of `features`'s

        Args:
            features (dict): (name, value) pairs

        Returns:
           (bool): True if superset relationship holds else False
        """
        return all([self.data[k] == v for (k, v) in ft_mask.items()])

    def __ge__(self, other):
        """Determine whether `self`'s features are a superset of `other`'s"""
        return self.match(other)

    def intersection(self, other):
        """Return dict of features shared by `self` and `other`

        Args:
            other (Segment): object with feature specifications

        Returns:
            Segment: (name, value) pairs for each shared feature
        """
        data = dict(set(self.items()) & set(other.items()))
        names = list(filter(lambda a: a in data, self.names))
        return Segment(names, data)

    def __and__(self, other):
        """Return dict of features shared by `self` and `other`"""
        return self.intersection(other)

    def numeric(self, names=None):
        if not names:
            names = self.names
        """Return feature values as a list of integers"""
        return [self.data[k] for k in names]

    def strings(self, names=None):
        """Return feature values as a list of strings"""
        if not names:
            names = self.names
        return list(map(lambda x: self.n2s[x], self.numeric()))

    def distance(self, other):
        """Compute a distance between `self` and `other`

        Args:
            other (Segment): object to compare with `self`

        Returns:
            int: the sum of the absolute value of the difference between each
                 of the feature values in `self` and `other`.
        """
        return sum(abs(a - b) for (a, b) in zip(self.numeric(), other.numeric()))

    def norm_distance(self, other):
        """Compute a distance, normalized by vector length

        Args:
            other (Segment): object to compare with `self`

        Returns:
            float: the sum of the absolute value of the difference between
                   each of the feature values in `self` and `other`, divided
                   by the number of features per vector.
        """
        return self.distance(other) / len(self.names)

    def __sub__(self, other):
        """Distance between segments, normalized by vector length"""
        return self.norm_distance(other)

    def hamming_distance(self, other):
        """Compute Hamming distance between feature vectors

        Args:
            other (Segment): object to compare with `self`

        Returns:
            int: the unnormalized Hamming distance between the two vectors.
        """
        return sum(int(a != b) for (a, b) in zip(self.numeric(), other.numeric()))

    def norm_hamming_distance(self, other):
        """Compute Hamming distance, normalized by vector length

        Args:
            other (Segment): object to compare with `self`

        Returns:
            int: the normalized Hamming distance between the two vectors.
        """
        return self.hamming_distance(other) / len(self.names)

    def weighted_distance(self, other):
        """Compute weighted distance

        Args:
            other (Segment): object to compare with `self`

        Returns:
            float: the weighted distance between the two vectors
        """
        return sum([abs(a - b) * c for (a, b, c)
                   in zip(self.numeric(), other.numeric(), self.weights)])

    def norm_weighted_distance(self, other):
        """Compute weighted distance, normalized by vector length

        Args:
            other (Segment): object to compare with `self`

        Returns:
            float: the weighted distance between the two vectors, normalized by
                   vector length.
        """
        return self.weighted_distance(other) / sum(self.weights)

    def specified(self):
        """Return dictionary of features that are specified '+' or '-' (1 or -1)

        Returns:
            dict: each feature in `self` for which the value is not 0
        """
        return {k: v for (k, v) in self.data.items() if v != 0}

    def differing_specs(self, other):
        """Return a list of feature names that differ in their specified values

        Args:
            other (Segment): object to compare with `self`

        Returns:
            list: the names of the features that differ in the two vectors
        """
        return [k for (k, v) in self.items() if other[k] != v]
