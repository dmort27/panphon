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

    def test_sonority_one_segment(self):
        self.assertEqual(self.ft.sonority(u'a'), 9)

    def test_sonority_basic(self):
        segs = [u'a', u'e', u'j', u'l', u'n', u'ʒ', u's', u'b', u't', ]
        scores = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        self.assertEqual(map(self.ft.sonority, segs), scores)

if __name__ == '__main__':
    unittest.main()
