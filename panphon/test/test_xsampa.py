# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import

import unittest
import panphon
import panphon.xsampa


class TestXSampa(unittest.TestCase):

    def setUp(self):
        """
        Sets the feature.

        Args:
            self: (todo): write your description
        """
        self.ft = panphon.FeatureTable()
        self.xs = panphon.xsampa.XSampa()

    def test_ipa_equals_xsampa(self):
        """
        Test if the ipa ipa is equal.

        Args:
            self: (todo): write your description
        """
        self.assertEqual('kʰat', self.xs.convert('k_h a t'))

    def test_ipa_vector_equals_xsampa_vector(self):
        """
        Compute the ipa ipa vectors.

        Args:
            self: (todo): write your description
        """
        ipa = self.ft.word_to_vector_list('kʰat', xsampa=False)
        xs = self.ft.word_to_vector_list('k_h a t', xsampa=True)
        self.assertEqual(ipa, xs)
