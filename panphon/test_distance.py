#!//usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division

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


class TestUnweightedFeatureEditDist(unittest.TestCase):
    def setUp(self):
        self.dist = distance.Distance()

    def test_unweighted_substitution_cost(self):
        self.assertEqual(self.dist.unweighted_substitution_cost(['0', '+', '-'], ['0', '+', '+']) * 3, 1)

    def test_unweighted_deletion_cost(self):
        self.assertEqual(self.dist.unweighted_deletion_cost(['+', '-', '+', '0']) * 4, 3.5)

    def test_trivial1(self):
        self.assertEqual(self.dist.feature_edit_distance('bim', 'pym') * 21, 3)

    def test_trivial2(self):
        self.assertEqual(self.dist.feature_edit_distance('ti', 'tʰi') * 21, 1)


class TestWeightedFeatureEditDist(unittest.TestCase):
    def setUp(self):
        self.dist = distance.Distance()

    def test_trivial1(self):
        self.assertGreater(self.dist.weighted_feature_edit_distance('ti', 'tʰu'),
                           self.dist.weighted_feature_edit_distance('ti', 'tʰi'))

    def test_trivial2(self):
        self.assertGreater(self.dist.weighted_feature_edit_distance('ti', 'te'),
                           self.dist.weighted_feature_edit_distance('ti', 'tḭ'))


class TestHammingFeatureEditDistanceDivMaxlen(unittest.TestCase):
    def setUp(self):
        self.dist = distance.Distance()

    def test_hamming_substitution_cost(self):
        self.assertEqual(self.dist.hamming_substitution_cost(['+', '-', '0'], ['0', '-', '0']) * 3, 1)

    def test_trivial1(self):
        self.assertEqual(self.dist.hamming_feature_edit_distance_div_maxlen('pa', 'ba') * 21 * 2, 1)

    def test_trivial2(self):
        self.assertEqual(self.dist.hamming_feature_edit_distance_div_maxlen('i', 'pi') * 2, 1)

    def test_trivial3(self):
        self.assertEqual(self.dist.hamming_feature_edit_distance_div_maxlen('sɛks', 'ɛɡz'), (1 + (1 / 21) + (1 / 21)) / 4)

    def test_trivial4(self):
        self.assertEqual(self.dist.hamming_feature_edit_distance_div_maxlen('k', 'ɡ'), 1 / 21)
