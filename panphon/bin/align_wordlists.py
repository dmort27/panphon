#!/usr/bin/env python
from __future__ import print_function

import unicodecsv as csv
import argparse
import panphon
import Levenshtein
import munkres
from functools import partial


def levenshtein_dist(_, a, b):
    return Levenshtein.distance(a, b)


def dogol_leven_dist(_, a, b):
    return Levenshtein.distance(ft.map_to_dogol_prime(a),
                                ft.map_to_dogol_prime(b))


def feature_hamming_dist(ft, a, b):
    return ft.feature_edit_distance(ft.word_to_vector_list(a),
                                    ft.word_to_vector_list(b))


def feature_weighted_dist(ft, a, b):
    return ft.weighted_feature_edit_distance(ft.word_to_vector_list(a),
                                             ft.word_to_vector_list(b))


def construct_cost_matrix(words_a, words_b, dist):
    def matrix_row(word_a, words_b):
        return [dist(word_a, word_b) for (word_b, _) in words_b]
    return [matrix_row(word_a, words_b) for (word_a, _) in words_a]


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
        print('Reading word lists...')
        words = zip([(w, g) for (g, w) in reader_a],
                    [(w, g) for (g, w) in reader_b])
        words_a, words_b = zip(*[(a, b) for (a, b) in words if a and b])
        print('Constructing cost matrix...')
        matrix = construct_cost_matrix(words_a, words_b, dist)
        m = munkres.Munkres()
        print('Computing matrix using Hungarian Algorithm...')
        indices = m.compute(matrix)
        print(score(indices))
        print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='Align two lists of "cognates" using a specified distance metric.')
    parser.add_argument('wordlists', nargs=2, help='Filenames of two wordlists in corresponding order.')
    parser.add_argument('-d', '--dist', default='hamming', help='Distance metric (e.g. Hamming).')
    args = parser.parse_args()
    dists = {'levenshtein': levenshtein_dist,
             'dogol_leven': dogol_leven_dist,
             'hamming': feature_hamming_dist,
             'weighted': feature_weighted_dist}
    ft = panphon.FeatureTable()
    dist = partial(dists[args.dist], ft)
    main(args.wordlists[0], args.wordlists[1], dist)
