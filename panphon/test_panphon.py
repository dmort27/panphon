# -*- coding: utf-8 -*-

import unittest
import _panphon


class TestIpaRe(unittest.TestCase):

    def setUp(self):
        self.ft = _panphon.FeatureTable()

    def test_compile_regex_from_str1(self):
        r = self.ft.compile_regex_from_str('[-son -cont][+syl -hi -lo]')
        self.assertIsNotNone(r.match(u'tʰe'))
        self.assertIsNone(r.match(u'pi'))

    def test_compile_regex_from_str2(self):
        r = self.ft.compile_regex_from_str(u'[-son -cont][+son +cont]')
        self.assertIsNotNone(r.match(u'pj'))
        self.assertIsNone(r.match(u'ts'))


class TestSonority(unittest.TestCase):

    def setUp(self):
        self.ft = _panphon.FeatureTable()

    def test_sonority_nine(self):
        segs = [u'a', u'ɑ', u'æ', u'ɒ', u'e', u'o']
        scores = [9] * 6
        self.assertEqual(map(self.ft.sonority, segs), scores)

    def test_sonority_eight(self):
        segs = [u'i', u'u', u'ɨ', u'ʉ', u'ɯ', u'u']
        scores = [8] * 6
        self.assertEqual(map(self.ft.sonority, segs), scores)

    def test_sonority_seven(self):
        segs = [u'j', u'w', u'ʋ', u'ɰ', u'ɹ']
        scores = [7] * 5
        self.assertEqual(map(self.ft.sonority, segs), scores)

    def test_sonority_six(self):
        segs = [u'l', u'ɭ', u'r', u'ɾ']
        scores = [6] * 4
        self.assertEqual(map(self.ft.sonority, segs), scores)

    def test_sonority_five(self):
        segs = [u'n', u'm', u'ŋ', u'ɴ']
        scores = [5] * 4
        self.assertEqual(map(self.ft.sonority, segs), scores)

    def test_sonority_four(self):
        segs = [u'v', u'z', u'ʒ', u'ɣ']
        scores = [4] * 4
        self.assertEqual(map(self.ft.sonority, segs), scores)

    def test_sonority_three(self):
        segs = [u'f', u's', u'x', u'ħ', u'ʃ']
        scores = [3] * 5
        self.assertEqual(map(self.ft.sonority, segs), scores)

    def test_sonority_two(self):
        segs = [u'b', u'g', u'd', u'ɢ']
        scores = [2] * 4
        self.assertEqual(map(self.ft.sonority, segs), scores)

    def test_sonority_one(self):
        segs = [u'p', u'k', u'c', u'q']
        scores = [1] * 4
        self.assertEqual(map(self.ft.sonority, segs), scores)

if __name__ == '__main__':
    unittest.main()
