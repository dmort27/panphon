# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import

import unittest
from panphon import permissive


class TestFeatureTableAPI(unittest.TestCase):

    def setUp(self):
        """
        Sets the list of this feature.

        Args:
            self: (todo): write your description
        """
        self.ft = permissive.PermissiveFeatureTable()

    def test_fts(self):
        """
        Test if the test test.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(len(self.ft.fts('u')), 22)

    # def test_seg_fts(self):
    #     self.assertEqual(len(self.ft.seg_fts('p')), 21)

    def test_match(self):
        """
        Matches the test.

        Args:
            self: (todo): write your description
        """
        self.assertTrue(self.ft.match(self.ft.fts('u'), self.ft.fts('u')))

    def test_fts_match(self):
        """
        Check if the match match match.

        Args:
            self: (todo): write your description
        """
        self.assertTrue(self.ft.fts_match(self.ft.fts('u'), 'u'))

    def test_longest_one_seg_prefix(self):
        """
        Set the longest longest prefix.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.ft.longest_one_seg_prefix('pap'), 'p')

    def test_validate_word(self):
        """
        Check that the word is valid.

        Args:
            self: (todo): write your description
        """
        self.assertTrue(self.ft.validate_word('tik'))

    def test_segs(self):
        """
        Check if the segs : segs : none

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.ft.segs('tik'), ['t', 'i', 'k'])

    def test_word_fts(self):
        """
        Test if a word.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(len(self.ft.word_fts('tik')), 3)

    def test_seg_known(self):
        """
        Check if the seg seg seg seg segment.

        Args:
            self: (todo): write your description
        """
        self.assertTrue(self.ft.seg_known('t'))

    def test_filter_string(self):
        """
        Filter the filter filter string.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(len(self.ft.filter_string('pup$')), 3)

    def test_segs_safe(self):
        """
        Check if the segs is a valid.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(len(self.ft.segs_safe('pup$')), 4)

    def test_filter_segs(self):
        """
        Filter out the segs : none

        Args:
            self: (todo): write your description
        """
        self.assertEqual(len(self.ft.filter_segs(['p', 'u', 'p', '$'])), 3)

    def test_fts_intersection(self):
        """
        Determine the intersection between two sets.

        Args:
            self: (todo): write your description
        """
        self.assertIn(('-', 'voi'), self.ft.fts_intersection(['p', 't', 'k']))

    def test_fts_match_any(self):
        """
        Check if any test features that match are included features.

        Args:
            self: (todo): write your description
        """
        self.assertTrue(self.ft.fts_match_any([('-', 'voi')], ['p', 'o', '$']))

    def test_fts_match_all(self):
        """
        Test if all features in the database.

        Args:
            self: (todo): write your description
        """
        self.assertTrue(self.ft.fts_match_all([('-', 'voi')], ['p', 't', 'k']))

    def test_fts_contrast2(self):
        """
        Test if the features in the features.

        Args:
            self: (todo): write your description
        """
        self.assertTrue(self.ft.fts_contrast2([], 'voi', ['p', 'b', 'r']))

    def test_fts_count(self):
        """
        Deter count is_fts_count coefficients.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(self.ft.fts_count([('-', 'voi')], ['p', 't', 'k', 'r']), 3)
        self.assertEqual(self.ft.fts_count([('-', 'voi')], ['r', '$']), 0)

    def test_match_pattern(self):
        """
        Matches a match.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(len(self.ft.match_pattern([set([('-', 'voi')])], 'p')), 1)

    def test_match_pattern_seq(self):
        """
        Matches the test sequence.

        Args:
            self: (todo): write your description
        """
        self.assertTrue(self.ft.match_pattern_seq([set([('-', 'voi')])], 'p'))

    # def test_all_segs_matching_fts(self):
    #     self.assertIn('p', self.ft.all_segs_matching_fts([('-', 'voi')]))

    def test_compile_regex_from_str(self):
        """
        Test if a regex string.

        Args:
            self: (todo): write your description
        """
        pass

    def test_segment_to_vector(self):
        """
        Test if the segment segment to segment.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(len(self.ft.segment_to_vector('p')), 22)

    def test_word_to_vector_list(self):
        """
        Convert list of vectors to list.

        Args:
            self: (todo): write your description
        """
        self.assertEqual(len(self.ft.word_to_vector_list('pup')), 3)
