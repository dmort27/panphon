#!//usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import

import unittest
from panphon import distance

feature_model = 'permissive'


class TestLevenshtein(unittest.TestCase):
    def setUp(self):
        self.dist = distance.Distance(feature_model=feature_model)

    def test_trivial1(self):
        self.assertEqual(self.dist.levenshtein_distance('pop', 'pʰop'), 1)

    def test_trivial2(self):
        self.assertEqual(self.dist.levenshtein_distance('pop', 'pʰom'), 2)


class TestDogolPrime(unittest.TestCase):
    def setUp(self):
        self.dist = distance.Distance(feature_model=feature_model)

    def test_trivial1(self):
        self.assertEqual(self.dist.dogol_prime_distance('pop', 'bob'), 0)

    def test_trivial2(self):
        self.assertEqual(self.dist.dogol_prime_distance('pop', 'bab'), 0)


class TestUnweightedFeatureEditDist(unittest.TestCase):
    def setUp(self):
        self.dist = distance.Distance(feature_model=feature_model)

    def test_unweighted_substitution_cost(self):
        self.assertEqual(self.dist.unweighted_substitution_cost(['0', '+', '-'], ['0', '+', '+']) * 3, 1)

    def test_unweighted_deletion_cost(self):
        self.assertEqual(self.dist.unweighted_deletion_cost(['+', '-', '+', '0']) * 4, 3.5)

    def test_trivial1(self):
        self.assertEqual(self.dist.feature_edit_distance('bim', 'pym') * 22, 3)

    def test_trivial2(self):
        self.assertEqual(self.dist.feature_edit_distance('ti', 'tʰi') * 22, 1)


class TestWeightedFeatureEditDist(unittest.TestCase):
    def setUp(self):
        self.dist = distance.Distance(feature_model=feature_model)

    def test_trivial1(self):
        self.assertGreater(self.dist.weighted_feature_edit_distance('ti', 'tʰu'),
                           self.dist.weighted_feature_edit_distance('ti', 'tʰi'))

    def test_trivial2(self):
        self.assertGreater(self.dist.weighted_feature_edit_distance('ti', 'te'),
                           self.dist.weighted_feature_edit_distance('ti', 'tḭ'))


class TestHammingFeatureEditDistanceDivMaxlen(unittest.TestCase):
    def setUp(self):
        self.dist = distance.Distance(feature_model=feature_model)

    def test_hamming_substitution_cost(self):
        self.assertEqual(self.dist.hamming_substitution_cost(['+', '-', '0'], ['0', '-', '0']) * 3, 1)

    def test_trivial1(self):
        self.assertEqual(self.dist.hamming_feature_edit_distance_div_maxlen('pa', 'ba') * 22 * 2, 1)

    def test_trivial2(self):
        self.assertEqual(self.dist.hamming_feature_edit_distance_div_maxlen('i', 'pi') * 2, 1)

    def test_trivial3(self):
        self.assertEqual(self.dist.hamming_feature_edit_distance_div_maxlen('sɛks', 'ɛɡz'), (1 + (1 / 22) + (1 / 22)) / 4)

    def test_trivial4(self):
        self.assertEqual(self.dist.hamming_feature_edit_distance_div_maxlen('k', 'ɡ'), 1 / 22)


class TestMany(unittest.TestCase):
    def setUp(self):
        self.dist = distance.Distance(feature_model=feature_model)

    def test_fast_levenshtein_distance(self):
        self.assertEqual(self.dist.fast_levenshtein_distance('p', 'b'), 1)

    def test_fast_levenshtein_distance_div_maxlen(self):
        self.assertEqual(self.dist.fast_levenshtein_distance_div_maxlen('p', 'b'), 1)

    def test_dogol_prime_distance(self):
        self.assertEqual(self.dist.dogol_prime_distance('p', 'b'), 0)

    def test_dogol_prime_div_maxlen(self):
        self.assertEqual(self.dist.dogol_prime_distance_div_by_maxlen('p', 'b'), 0)

    def test_feature_edit_distance(self):
        self.assertEqual(self.dist.feature_edit_distance('p', 'b'), 1 / 22)

    def test_jt_feature_edit_distance(self):
        self.assertEqual(self.dist.jt_feature_edit_distance('p', 'b'), 1 / 22)

    def test_feature_edit_distance_div_by_maxlen(self):
        self.assertEqual(self.dist.feature_edit_distance_div_by_maxlen('p', 'b'), 1 / 22)

    def test_jt_feature_edit_distance_div_by_maxlen(self):
        self.assertEqual(self.dist.jt_feature_edit_distance_div_by_maxlen('p', 'b'), 1 / 22)

    def test_hamming_feature_edit_distance(self):
        self.assertEqual(self.dist.hamming_feature_edit_distance('p', 'b'), 1 / 22)

    def test_jt_hamming_feature_edit_distance(self):
        self.assertEqual(self.dist.jt_hamming_feature_edit_distance('p', 'b'), 1 / 22)

    def test_hamming_feature_edit_distance_div_maxlen(self):
        self.assertEqual(self.dist.hamming_feature_edit_distance_div_maxlen('p', 'b'), 1 / 22)

    def test_jt_hamming_feature_edit_distance_div_maxlen(self):
        self.assertEqual(self.dist.jt_hamming_feature_edit_distance_div_maxlen('p', 'b'), 1 / 22)

    def test_weighted_feature_edit_distance(self):
        self.assertEqual(self.dist.weighted_feature_edit_distance('p', 'b'), 1 / 8)

    def test_weighted_feature_edit_distance_div_maxlen(self):
        self.assertEqual(self.dist.weighted_feature_edit_distance_div_maxlen('p', 'b'), 1 / 8)


class TestXSampa(unittest.TestCase):
    def setUp(self):
        self.dist = distance.Distance(feature_model=feature_model)

    def test_feature_edit_distance(self):
        self.assertEqual(self.dist.feature_edit_distance("p_h", "p", xsampa=True), 1 / 22)
