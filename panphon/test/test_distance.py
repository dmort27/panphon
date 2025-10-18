# -*- coding: utf-8 -*-

import unittest
import panphon
from panphon import distance

feature_model = 'segment'
dim = 24


class TestLevenshtein(unittest.TestCase):
    def setUp(self):
        self.dist = distance.Distance(feature_model=feature_model)

    def test_trivial1(self):
        self.assertEqual(self.dist.levenshtein_distance('pop', 'pʰop'), 1)

    def test_trivial2(self):
        self.assertEqual(self.dist.levenshtein_distance('pop', 'pʰom'), 2)


class TestDolgoPrime(unittest.TestCase):
    def setUp(self):
        self.dist = distance.Distance(feature_model=feature_model)

    def test_trivial1(self):
        self.assertEqual(self.dist.dolgo_prime_distance('pop', 'bob'), 0)

    def test_trivial2(self):
        self.assertEqual(self.dist.dolgo_prime_distance('pop', 'bab'), 0)


class TestUnweightedFeatureEditDist(unittest.TestCase):
    def setUp(self):
        self.dist = distance.Distance(feature_model=feature_model)

    def test_unweighted_substitution_cost(self):
        self.assertEqual(self.dist.unweighted_substitution_cost([0, 1, -1], [0, 1, 1]) * 3, 1)

    def test_unweighted_deletion_cost(self):
        self.assertEqual(self.dist.unweighted_deletion_cost([1, -1, 1, 0]) * 4, 3.5)

    def test_trivial1(self):
        self.assertEqual(self.dist.feature_edit_distance('bim', 'pym') * dim, 3)

    def test_trivial2(self):
        self.assertEqual(self.dist.feature_edit_distance('ti', 'tʰi') * dim, 1)

    def test_xsampa(self):
        self.assertEqual(self.dist.feature_edit_distance('t i', 't_h i', xsampa=True) * dim, 1)

    def test_xsampa2(self):
        self.assertEqual(self.dist.feature_edit_distance('p u n', 'p y n', xsampa=True) * dim, 1)

    def test_xsampa3(self):
        ipa = self.dist.jt_feature_edit_distance_div_maxlen('kʰin', 'pʰin')
        xs = self.dist.jt_feature_edit_distance_div_maxlen('k_h i n', 'p_h i n', xsampa=True)
        self.assertEqual(ipa, xs)


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
        self.assertEqual(self.dist.hamming_feature_edit_distance_div_maxlen('pa', 'ba') * dim * 2, 1)

    def test_trivial2(self):
        self.assertEqual(self.dist.hamming_feature_edit_distance_div_maxlen('i', 'pi') * 2, 1)

    def test_trivial3(self):
        self.assertEqual(self.dist.hamming_feature_edit_distance_div_maxlen('sɛks', 'ɛɡz'), (1 + (1 / dim) + (1 / dim)) / 4)

    def test_trivial4(self):
        self.assertEqual(self.dist.hamming_feature_edit_distance_div_maxlen('k', 'ɡ'), 1 / dim)


class TestMany(unittest.TestCase):
    def setUp(self):
        self.dist = distance.Distance(feature_model=feature_model)

    def test_fast_levenshtein_distance(self):
        self.assertEqual(self.dist.fast_levenshtein_distance('p', 'b'), 1)

    def test_fast_levenshtein_distance_div_maxlen(self):
        self.assertEqual(self.dist.fast_levenshtein_distance_div_maxlen('p', 'b'), 1)

    def test_dolgo_prime_distance(self):
        self.assertEqual(self.dist.dolgo_prime_distance('p', 'b'), 0)

    def test_dolgo_prime_div_maxlen(self):
        self.assertEqual(self.dist.dolgo_prime_distance_div_maxlen('p', 'b'), 0)

    def test_feature_edit_distance(self):
        self.assertEqual(self.dist.feature_edit_distance('p', 'b'), 1 / dim)

    def test_jt_feature_edit_distance(self):
        self.assertEqual(self.dist.jt_feature_edit_distance('p', 'b'), 1 / dim)

    def test_feature_edit_distance_div_maxlen(self):
        self.assertEqual(self.dist.feature_edit_distance_div_maxlen('p', 'b'), 1 / dim)

    def test_jt_feature_edit_distance_div_maxlen(self):
        self.assertEqual(self.dist.jt_feature_edit_distance_div_maxlen('p', 'b'), 1 / dim)

    def test_hamming_feature_edit_distance(self):
        self.assertEqual(self.dist.hamming_feature_edit_distance('p', 'b'), 1 / dim)

    def test_jt_hamming_feature_edit_distance(self):
        self.assertEqual(self.dist.jt_hamming_feature_edit_distance('p', 'b'), 1 / dim)

    def test_hamming_feature_edit_distance_div_maxlen(self):
        self.assertEqual(self.dist.hamming_feature_edit_distance_div_maxlen('p', 'b'), 1 / dim)

    def test_jt_hamming_feature_edit_distance_div_maxlen(self):
        self.assertEqual(self.dist.jt_hamming_feature_edit_distance_div_maxlen('p', 'b'), 1 / dim)

class TestXSampa(unittest.TestCase):
    def setUp(self):
        self.dist = distance.Distance(feature_model=feature_model)
        self.ft = panphon.FeatureTable()

    def test_feature_edit_distance(self):
        self.assertEqual(self.dist.feature_edit_distance("p_h", "p", xsampa=True), 1 / dim)
