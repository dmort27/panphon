# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import

import unittest
import panphon
from panphon import distance

feature_model = 'segment'


class TestLevenshtein(unittest.TestCase):
    def setUp(self):
        """
        Sets the distance between the distance.

        Args:
            self: (todo): write your description
        """
        self.dist = distance.Distance(feature_model=feature_model)

    def test_trivial1(self):
        """
        Compute the distance between two levenshtein distance.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.levenshtein_distance('pop', 'pʰop'), 1)

    def test_trivial2(self):
        """
        Compute the levenshtein distance between 2nd distance.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.levenshtein_distance('pop', 'pʰom'), 2)


class TestDogolPrime(unittest.TestCase):
    def setUp(self):
        """
        Sets the distance between the distance.

        Args:
            self: (todo): write your description
        """
        self.dist = distance.Distance(feature_model=feature_model)

    def test_trivial1(self):
        """
        Calculates the distance between self.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.dogol_prime_distance('pop', 'bob'), 0)

    def test_trivial2(self):
        """
        Compute equalial distance between 2 - dimensional distance

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.dogol_prime_distance('pop', 'bab'), 0)


class TestUnweightedFeatureEditDist(unittest.TestCase):
    def setUp(self):
        """
        Sets the distance between the distance.

        Args:
            self: (todo): write your description
        """
        self.dist = distance.Distance(feature_model=feature_model)

    def test_unweighted_substitution_cost(self):
        """
        The cost of the cost.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.unweighted_substitution_cost([0, 1, -1], [0, 1, 1]) * 3, 1)

    def test_unweighted_deletion_cost(self):
        """
        Test the cost cost.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.unweighted_deletion_cost([1, -1, 1, 0]) * 4, 3.5)

    def test_trivial1(self):
        """
        Compute the distance between two feature

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.feature_edit_distance('bim', 'pym') * 22, 3)

    def test_trivial2(self):
        """
        Computes the distance between two feature

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.feature_edit_distance('ti', 'tʰi') * 22, 1)

    def test_xsampa(self):
        """
        Calculates distance between feature and feature

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.feature_edit_distance('t i', 't_h i', xsampa=True) * 22, 1)

    def test_xsampa2(self):
        """
        Compute the distance between two feature vectors.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.feature_edit_distance('p u n', 'p y n', xsampa=True) * 22, 1)

    def test_xsampa3(self):
        """
        Compute the distance between two points.

        Args:
            self: (todo): write your description
        """
        ipa = self.dist.jt_feature_edit_distance_div_maxlen('kʰin', 'pʰin')
        xs = self.dist.jt_feature_edit_distance_div_maxlen('k_h i n', 'p_h i n', xsampa=True)
        self.assertEqual(ipa, xs)


class TestWeightedFeatureEditDist(unittest.TestCase):
    def setUp(self):
        """
        Sets the distance between the distance.

        Args:
            self: (todo): write your description
        """
        self.dist = distance.Distance(feature_model=feature_model)

    def test_trivial1(self):
        """
        Edit distance between two distance between two features

        Args:
            self: (todo): write your description
        """
        self.assertGreater(self.dist.weighted_feature_edit_distance('ti', 'tʰu'),
                           self.dist.weighted_feature_edit_distance('ti', 'tʰi'))

    def test_trivial2(self):
        """
        Set feature distance between feature and feature

        Args:
            self: (todo): write your description
        """
        self.assertGreater(self.dist.weighted_feature_edit_distance('ti', 'te'),
                           self.dist.weighted_feature_edit_distance('ti', 'tḭ'))


class TestHammingFeatureEditDistanceDivMaxlen(unittest.TestCase):
    def setUp(self):
        """
        Sets the distance between the distance.

        Args:
            self: (todo): write your description
        """
        self.dist = distance.Distance(feature_model=feature_model)

    def test_hamming_substitution_cost(self):
        """
        Calculate hamming cost cost.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.hamming_substitution_cost(['+', '-', '0'], ['0', '-', '0']) * 3, 1)

    def test_trivial1(self):
        """
        Compute the distance between two features.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.hamming_feature_edit_distance_div_maxlen('pa', 'ba') * 22 * 2, 1)

    def test_trivial2(self):
        """
        Compute distance between two features.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.hamming_feature_edit_distance_div_maxlen('i', 'pi') * 2, 1)

    def test_trivial3(self):
        """
        Compute the equalial distance.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.hamming_feature_edit_distance_div_maxlen('sɛks', 'ɛɡz'), (1 + (1 / 22) + (1 / 22)) / 4)

    def test_trivial4(self):
        """
        Test to true / gqualial features.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.hamming_feature_edit_distance_div_maxlen('k', 'ɡ'), 1 / 22)


class TestMany(unittest.TestCase):
    def setUp(self):
        """
        Sets the distance between the distance.

        Args:
            self: (todo): write your description
        """
        self.dist = distance.Distance(feature_model=feature_model)

    def test_fast_levenshtein_distance(self):
        """
        Test for levenshtein distance.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.fast_levenshtein_distance('p', 'b'), 1)

    def test_fast_levenshtein_distance_div_maxlen(self):
        """
        The levenshtein distance between the minimum distance.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.fast_levenshtein_distance_div_maxlen('p', 'b'), 1)

    def test_dogol_prime_distance(self):
        """
        Assigns the distance between self. hqual_prime

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.dogol_prime_distance('p', 'b'), 0)

    def test_dogol_prime_div_maxlen(self):
        """
        Determine the maximum number of the maximum length of the minimum.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.dogol_prime_distance_div_maxlen('p', 'b'), 0)

    def test_feature_edit_distance(self):
        """
        Set feature distance

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.feature_edit_distance('p', 'b'), 1 / 22)

    def test_jt_feature_edit_distance(self):
        """
        Test for jt distance

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.jt_feature_edit_distance('p', 'b'), 1 / 22)

    def test_feature_edit_distance_div_maxlen(self):
        """
        The feature feature feature feature feature distance.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.feature_edit_distance_div_maxlen('p', 'b'), 1 / 22)

    def test_jt_feature_edit_distance_div_maxlen(self):
        """
        Returns the feature feature feature feature feature feature.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.jt_feature_edit_distance_div_maxlen('p', 'b'), 1 / 22)

    def test_hamming_feature_edit_distance(self):
        """
        Test that the hamming distance.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.hamming_feature_edit_distance('p', 'b'), 1 / 22)

    def test_jt_hamming_feature_edit_distance(self):
        """
        Compute jt jt jt jt jt distance.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.jt_hamming_feature_edit_distance('p', 'b'), 1 / 22)

    def test_hamming_feature_edit_distance_div_maxlen(self):
        """
        Calculate the hamming distance.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.hamming_feature_edit_distance_div_maxlen('p', 'b'), 1 / 22)

    def test_jt_hamming_feature_edit_distance_div_maxlen(self):
        """
        Calculate the jt jt jt jt distance.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.jt_hamming_feature_edit_distance_div_maxlen('p', 'b'), 1 / 22)

    def test_weighted_feature_edit_distance(self):
        """
        Updates the distance

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.weighted_feature_edit_distance('p', 'b'), 1 / 8)

    def test_weighted_feature_edit_distance_div_maxlen(self):
        """
        Calculate feature distance

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.weighted_feature_edit_distance_div_maxlen('p', 'b'), 1 / 8)


class TestXSampa(unittest.TestCase):
    def setUp(self):
        """
        Set the distance between the featuremodel.

        Args:
            self: (todo): write your description
        """
        self.dist = distance.Distance(feature_model=feature_model)
        self.ft = panphon.FeatureTable()

    def test_feature_edit_distance(self):
        """
        Test if feature distance

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.dist.feature_edit_distance("p_h", "p", xsampa=True), 1 / 22)
