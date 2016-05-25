#!//usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals

import unittest
import distance


class TestLevenshtein(unittest.TestCase):
    def setUp(self):
        self.dist = distance.Distance()

    def test_trivial1(self):
        self.assertEqual(self.dist.levenshtein_distance('pop', 'pʰop'), 1)

    def test_trivial2(self):
        self.assertEqual(self.dist.levenshtein_distance('pop', 'pʰom'), 2)


class TestDogolPrime(unittest.TestCase):
    def setUp(self):
        self.dist = distance.Distance()

    def test_trivial1(self):
        self.assertEqual(self.dist.dogol_prime_distance('pop', 'bob'), 0)

    def test_trivial2(self):
        self.assertEqual(self.dist.dogol_prime_distance('pop', 'bab'), 0)


class TestUnweightedFtEditDist(unittest.TestCase):
    def setUp(self):
        self.dist = distance.Distance()

    def test_trivial1(self):
        self.assertEqual(self.dist.feature_edit_distance(self.dist.segs('pim'), self.dist.segs('bym')), 2)

    def test_trivial2(self):
        self.assertEqual(self.dist.feature_edit_distance(self.dist.segs('ti'), self.dist.segs('tʰi')), 1)
