# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import

import unittest
from panphon import sonority


class TestSonority(unittest.TestCase):

    def setUp(self):
        """
        Set the sonority.

        Args:
            self: (todo): write your description
        """
        self.son = sonority.Sonority(feature_model='permissive')

    def test_sonority_nine(self):
        """
        Generate sonority scores.

        Args:
            self: (todo): write your description
        """
        segs = ['a', 'ɑ', 'æ', 'ɒ', 'e', 'o̥']
        scores = [9] * 6
        self.assertEqual(list(map(self.son.sonority, segs)), scores)

    def test_sonority_eight(self):
        """
        Determine the sonority scores.

        Args:
            self: (todo): write your description
        """
        segs = ['i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u']
        scores = [8] * 6
        self.assertEqual(list(map(self.son.sonority, segs)), scores)

    def test_sonority_seven(self):
        """
        Test if a sonority scores.

        Args:
            self: (todo): write your description
        """
        segs = ['j', 'w', 'ʋ', 'ɰ', 'ɹ', 'e̯']
        scores = [7] * 6
        self.assertEqual(list(map(self.son.sonority, segs)), scores)

    def test_sonority_six(self):
        """
        Test if sonority scores are set.

        Args:
            self: (todo): write your description
        """
        segs = ['l', 'ɭ', 'r', 'ɾ']
        scores = [6] * 4
        self.assertEqual(list(map(self.son.sonority, segs)), scores)

    def test_sonority_five(self):
        """
        Test if there s a sonos.

        Args:
            self: (todo): write your description
        """
        segs = ['n', 'm', 'ŋ', 'ɴ']
        scores = [5] * 4
        self.assertEqual(list(map(self.son.sonority, segs)), scores)

    def test_sonority_four(self):
        """
        Test if sonority scores.

        Args:
            self: (todo): write your description
        """
        segs = ['v', 'z', 'ʒ', 'ɣ']
        scores = [4] * 4
        self.assertEqual(list(map(self.son.sonority, segs)), scores)

    def test_sonority_three(self):
        """
        Test if sonority of the number of the sonority.

        Args:
            self: (todo): write your description
        """
        segs = ['f', 's', 'x', 'ħ', 'ʃ']
        scores = [3] * 5
        self.assertEqual(list(map(self.son.sonority, segs)), scores)

    def test_sonority_two(self):
        """
        Test if two sonority scores.

        Args:
            self: (todo): write your description
        """
        segs = ['b', 'ɡ', 'd', 'ɢ']
        scores = [2] * 4
        self.assertEqual(list(map(self.son.sonority, segs)), scores)

    def test_sonority_one(self):
        """
        Test if there are two - qubit.

        Args:
            self: (todo): write your description
        """
        segs = ['p', 'k', 'c', 'q']
        scores = [1] * 4
        self.assertEqual(list(map(self.son.sonority, segs)), scores)
