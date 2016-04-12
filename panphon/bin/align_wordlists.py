#!/usr/bin/env python

import unicodecsv as csv
import argparse
import panphon
import Levenshtein
import munkres
from functools import partial


def levenshtein_dist(_, a, b):
    return Levenshtein.distance(a, b)


def feature_hamming_dist(ft, a, b):
    return ft.feature_edit_distance(ft.segment_to_vector(a),
                                    ft.segment_to_vector(b))


def feature_weighted_dist(ft, a, b):
    return ft.weighted_feature_edit_distance(ft.segment_to_vector(a),
                                             ft.segment_to_vector(b))


def construct_cost_matrix(words_a, words_b, dist):
    def matrix_row(word_a, words_b):
        return [dist(word_a, word_b) for (i, word_b, gloss_b) in words_b]
    return [matrix_row(word_a, words_b) for (i, word_a, gloss_a) in words_a]


def score(indices):
    pairs, errors = 0, 0
    for row, column in indices:
        pairs += 1
        if row != column:
            errors += 1
    return pairs, errors


def main(wordlist1, wordlist2, dist):
    with open(wordlist1, 'rb') as file_a, open(wordlist2, 'rb') as file_b:
        reader_a = csv.reader(file_a, encoding='utf-8')
        reader_b = csv.reader(file_b, encoding='utf-8')
        words_a = [(i, r[0], r[1]) for (i, r) in enumerate(reader_a)]
        words_b = [(i, r[0], r[1]) for (i, r) in enumerate(reader_b)]
        matrix = construct_cost_matrix(words_a, words_b, dist)
        m = munkres.Munkres()
        indices = m.compute(matrix)
        print score(indices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='Align two lists of "cognates" using a specified distance metric.')
    parser.add_argument('-w' '--wordlists', nargs=2, help='Filenames of two wordlists in corresponding order.')
    parser.add_argument('-d', '--dist', default='hamming', help='Distance metric (e.g. Hamming).')
    args = parser.parse_args()
    dists = {'levenshtein': levenshtein_dist,
             'hamming': feature_hamming_distance,
             'weighted': feature_weighted_dist}
    ft = panphon.FeatureTable()
    dist = partial(dists[args.dist], ft)
    main(args.wordlists[0], args.wordlists[1], dist)
