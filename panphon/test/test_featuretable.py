# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import

import unittest
import panphon.featuretable


class TestFeatureTable(unittest.TestCase):
    LONG_IPA_STRING = 'tɐʉmɐtɐ.ɸɐkɐtɐŋihɐŋɐ.koːɐʉɐʉ.ɔ.tɐmɐtɛɐ.tʉɾi.pʉkɐkɐ.piki.mɐʉŋɐ.hɔɾɔ.nʉkʉ.pɔkɐi.ɸɛnʉɐ.ki.tɐnɐ.tɐhʉ'
    TIMING_REPETITIONS = 1000

    def setUp(self):
        self.ft = panphon.featuretable.FeatureTable()

    def test_fts_contrast2(self):
        inv = 'p t k b d ɡ a e i o u'.split(' ')
        self.assertTrue(self.ft.fts_contrast({'syl': -1}, 'voi', inv))
        self.assertFalse(self.ft.fts_contrast({'syl': 1}, 'cor', inv))
        self.assertTrue(self.ft.fts_contrast({'ant': 1, 'cor': -1}, 'voi', inv))

    def test_longest_one_seg_prefix(self):
        prefix = self.ft.longest_one_seg_prefix('pʰʲaŋ')
        self.assertEqual(prefix, 'pʰʲ')

    def test_match_pattern(self):
        self.assertTrue(self.ft.match_pattern([{'voi': -1},
                                               {'voi': 1},
                                               {'voi': -1}],
                                              'pat'))

    def test_all_segs_matching_fts(self):
        segs = self.ft.all_segs_matching_fts({'syl': -1, 'son': 1})
        self.assertIn('m', segs)
        self.assertIn('n', segs)
        self.assertIn('ŋ', segs)
        self.assertIn('m̥', segs)
        self.assertIn('l', segs)
        # self.assertNotIn('a', segs)
        # self.assertNotIn('s', segs)

    def test_word_to_vector_list_aspiration(self):
        self.assertNotEqual(self.ft.word_to_vector_list(u'pʰ'),
                            self.ft.word_to_vector_list(u'p'))

    def test_word_to_vector_list_aspiration_xsampa(self):
        self.assertNotEqual(self.ft.word_to_vector_list(u'p_h', xsampa=True),
                            self.ft.word_to_vector_list(u'p', xsampa=True))

    def test_word_to_vector_list_aspiration_xsampa_len(self):
        self.assertEqual(len(self.ft.word_to_vector_list(u'p_h', xsampa=True)), 1)

    def test_normalization(self):
        u1 = '\u00e3'
        u2 = 'a\u0303'
        self.assertEqual(self.ft.ipa_segs(u1), self.ft.ipa_segs(u2))

    def test_ipa_segs_timing(self):
        for _ in range(self.TIMING_REPETITIONS):
            self.ft.ipa_segs(self.LONG_IPA_STRING)

    def test_segs_safe_timing(self):
        for _ in range(self.TIMING_REPETITIONS):
            self.ft.segs_safe(self.LONG_IPA_STRING)

class TestIpaRe(unittest.TestCase):

    def setUp(self):
        self.ft = panphon.featuretable.FeatureTable()

    def test_compile_regex_from_str1(self):
        r = self.ft.compile_regex_from_str('[-son -cont][+syl -hi -lo]')
        self.assertIsNotNone(r.match('tʰe'))
        self.assertIsNone(r.match('pi'))

    def test_compile_regex_from_str2(self):
        r = self.ft.compile_regex_from_str('[-son -cont][+son +cont]')
        self.assertIsNotNone(r.match('pj'))
        self.assertIsNone(r.match('ts'))

    def test_word_vector_list_english(self):
        input_word=u'maɪkɫbɛniætɡimeɪldɑtkʰɔm' # michaelbennie@gmail.com
        vector_list =self.ft.word_to_vector_list(input_word,numeric=True)
        new_word_list=self.ft.vector_list_to_word(vector_list)
        self.assertEqual(input_word,new_word_list)

    def test_word_vector_list_chinese(self):
        input_word=u'tɕi˥pʰu˧˥tʰɑʊ˧˥pu˥˩tʰu˩˨pʰu˧˥tʰɑʊ˧˥pʰi˧˥pu˥˩tɕi˥pʰu˧˥tʰɑʊ˧˥tɑʊ˥˩tʰu˩˨pʰu˧˥tʰɑʊ˧˥pʰi˧˥'
        vector_list =self.ft.word_to_vector_list(input_word,numeric=True)
        new_word_list=self.ft.vector_list_to_word(vector_list)
        self.assertEqual(input_word,new_word_list)


    def test_word_vector_list_french(self):
        input_word=u'ʒəkɔ̃pytəʁɛilzoʁɛkɔ̃pyte'
        vector_list =self.ft.word_to_vector_list(input_word,numeric=True)
        new_word_list=self.ft.vector_list_to_word(vector_list)
        self.assertEqual(input_word,new_word_list)


class TestXSampa(unittest.TestCase):

    def setUp(self):
        self.ft = panphon.featuretable.FeatureTable()

    def test_affricates(self):
        self.assertNotEqual(self.ft.word_to_vector_list(u'tS', xsampa=True),
                            self.ft.word_to_vector_list(u't S', xsampa=True))


if __name__ == '__main__':
    unittest.main()
