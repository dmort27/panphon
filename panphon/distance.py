from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
from functools import partial

import editdistance
import numpy as np
import pkg_resources
import yaml

from . import _panphon, permissive


class Distance(object):
    """Measures of phonological distance."""

    def __init__(self, feature_set='spe+', feature_model='strict'):
        """Construct a `Distance` object

        Args:
            feature_set (str): feature set to be used by the `Distance` object
            feature_model (str): feature parsing model to be used by the
                                 `Distance` object
        """
        filename = _panphon.filenames[feature_set]
        fm = {'strict': _panphon.FeatureTable,
              'permissive': permissive.PermissiveFeatureTable}
        self.fm = fm[feature_model](feature_set=feature_set)
        self.segments, self.seg_dict, self.names = self.fm._read_table(filename)
        self.weights = self.fm._read_weights()
        self.seg_regex = self.fm._build_seg_regex()
        self.longest_seg = max([len(x) for x in self.fm.seg_dict.keys()])
        self.dogol_prime = self._dogolpolsky_prime()

    def _dogolpolsky_prime(self, filename=os.path.join('data', 'dogolpolsky_prime.yml')):
        """Reads Dogolpolsky classes and constructs function cascade

        Args:
            filename (str): path to YAML file (from panphon root) containing
                            Dogolpolsky classes
        """
        filename = pkg_resources.resource_filename(
            __name__, filename)
        with open(filename, 'r') as f:
            rules = []
            dogol_prime = yaml.load(f.read())
            for rule in dogol_prime:
                rules.append((_panphon.fts(rule['def']), rule['label']))
        return rules

    def map_to_dogol_prime(self, s):
        """Map a string to Dogolpolsky' classes

        Args:
            s (unicode): IPA word

        Returns:
            s (unicode): word with all segments collapsed to D' classes
        """
        segs = []
        for seg in self.fm.seg_regex.findall(s):
            fts = self.fm.fts(seg)
            for mask, label in self.dogol_prime:
                if self.fm.match(mask, fts):
                    segs.append(label)
                    break
        return ''.join(segs)

    def feature_difference(self, ft1, ft2):
        """Given two feature values, return the numerical difference

        The difference between '+' and '-' is 1 and the difference between '0'
        and '+' or '-' is 0.5.

        Args:
            ft1 (str): feature value ('+', '-', or '0')
            ft2 (str): feature values ('+', '-', or '0')

        Returns:
            float: the difference between `ft1` and `ft2`:
                   '+' - '-' = 1.0
                   '-' - '+' = 1.0
                   '0' - '+' = 0.5
                   '0' - '-' = 0.5
                   '+' - '0' = 0.5
                   '-' - '0' = 0.5
        """
        tr = {'-': -1, '0': 0, '+': 1}
        return abs(tr[ft1] - tr[ft2]) / 2.0

    def levenshtein_distance(self, source, target):
        """Slow implementation of Levenshtein distance using NumPy arrays

        Args:
            source (unicode): source word
            target (unicode): target word

        Returns:
            int: minimum number of Levenshtein edits required to get from
                 `source` to `target`
        """
        if len(source) < len(target):
            return self.levenshtein_distance(target, source)
        # So now we have len(source) >= len(target).
        if len(target) == 0:
            return len(source)
        # We call tuple() to force strings to be used as sequences
        # ('c', 'a', 't', 's') - numpy uses them as values by default.
        source = np.array(tuple(source))
        target = np.array(tuple(target))
        # We use a dynamic programming algorithm, but with the
        # added optimization that we only need the last two rows
        # of the matrix.
        previous_row = np.arange(target.size + 1)
        for s in source:
            # Insertion (target grows longer than source):
            current_row = previous_row + 1
            # Substitution or matching:
            # Target and source items are aligned, and either
            # are different (cost of 1), or are the same (cost of 0).
            current_row[1:] = np.minimum(current_row[1:], np.add(previous_row[:-1], target != s))
            # Deletion (target grows shorter than source):
            current_row[1:] = np.minimum(current_row[1:], current_row[0:-1] + 1)
            previous_row = current_row
        return previous_row[-1]

    def fast_levenshtein_distance(self, source, target):
        """Wrapper for the distance function in the Levenshtein module.

        Args:
            source (unicode): source word
            target (unicode): target word

        Returns:
            int: minimum number of Levenshtein edits required to get from
                 `source` to `target`
        """
        return int(editdistance.eval(source, target))

    def dogol_prime_distance(self, source, target):
        """Levenshtein distance using D' phonetic equivalence classes

        `source` and `target` are converted to Dogolpolsky' equivalence classes
        (each segment is mapped to the appropriate class) and then the
        Levenshtein distance between the resulting representations is
        computed.

        Args:
            source (unicode): source word
            target (unicode): target word

        Returns:
            int: minimum number of Levenshtein edits required to get from
                 Dogolpolsky' versions of `source` to `target`
        """
        source = self.map_to_dogol_prime(source)
        target = self.map_to_dogol_prime(target)
        return self.fast_levenshtein_distance(source, target)

    def dogol_prime_distance_div_by_maxlen(self, source, target):
        """Levenshtein distance using D' classes, normalized by max length

        `source` and `target` are converted to Dogolpolsky' equivalence classes
        (each segment is mapped to the appropriate class) and then the
        Levenshtein distance between the resulting representations is
        computed. The result is divided by the length of the longest argument
        (`source` or `target`) after mapping to D' classes.

        Args:
            source (unicode): source word
            target (unicode): target word

        Returns:
            int: minimum number of Levenshtein edits required to get from
                 Dogolpolsky' versions of `source` to `target`
        """
        source = self.map_to_dogol_prime(source)
        target = self.map_to_dogol_prime(target)
        maxlen = max(len(source), len(target))
        return self.fast_levenshtein_distance(source, target) / maxlen

    def min_edit_distance(self, del_cost, ins_cost, sub_cost, start, source, target):
        """Return minimum edit distance, parameterized, slow

        Args:
            del_cost (function): cost function for deletion
            ins_cost (function): cost function for insertion
            sub_cost (function): cost function for substitution
            start (sequence): start symbol: string for strings, list for lists,
                              list of list for list of lists
            source (sequence): source string/sequence of feature vectors
            target (sequence): target string/sequence of feature vectors

        Returns:
            Number: minimum edit distance from source to target, with edit costs
                    as defined
        """
        # Get lengths of source and target
        n, m = len(source), len(target)
        source, target = start + source, start + target
        # Create "matrix"
        d = []
        for i in range(n + 1):
            d.append((m + 1) * [None])
        # Initialize "matrix"
        d[0][0] = 0
        for i in range(1, n + 1):
            d[i][0] = d[i - 1][0] + del_cost(source[i])
        for j in range(1, m + 1):
            d[0][j] = d[0][j - 1] + ins_cost(target[j])
        # Recurrence relation
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d[i][j] = min([
                    d[i - 1][j] + del_cost(source[i]),
                    d[i - 1][j - 1] + sub_cost(source[i], target[j]),
                    d[i][j - 1] + ins_cost(target[j]),
                ])
        return d[n][m]

    def unweighted_deletion_cost(self, v1, gl_wt=1.0):
        """Return cost of deleting segment corresponding to feature vector

        Features are not weighted; features specified as '0' add 0.5 to the raw
        deletion cost; other features add 1 to the raw deletion cost; the cost
        is normalized by the number of features

        Args:
            v1 (list): vector of feature values
            global_weight (Number): global weighting factor

        Returns:
            float: sum of feature costs divided by the number of features and
                   multiplied by a global weighting factor
        """
        assert isinstance(v1, list)
        return sum(map(lambda x: 0.5 if x == '0' else 1, v1)) / len(v1) * gl_wt

    def unweighted_substitution_cost(self, v1, v2):
        """Given two feature vectors, return the difference

        Args:
            v1 (list): vector of feature values
            v2 (list): vector of feature values

        Returns:
            float: sum of the differences between the features in `v1` and `v2`,
                   divided by the number of features
        """
        assert isinstance(v1, list)
        assert len(v1) == len(v2)
        diffs = [self.feature_difference(ft1, ft2)
                 for (ft1, ft2) in zip(v1, v2)]
        return sum(diffs) / len(v1)

    def unweighted_insertion_cost(self, v1, gl_wt=1.0):
        """Return cost of inserting segment corresponding to feature vector

        Features are not weighted; features with the value '0' add 0.5 to the
        raw cost; other features add 1.0 to the raw cost; the raw cost is then
        normalized by the number of features

        Args:
            v1 (list): vector of feature values
            global_weight (Number): global weighting factor

        Returns:
            float: sum of the costs of inserting each of the features in `v1`
                   divided by the number of features in the vector and
                   multiplied by a global weighting factor
        """
        assert isinstance(v1, list)
        return sum(map(lambda x: 0.5 if x == '0' else 1, v1)) / len(v1) * gl_wt

    def feature_edit_distance(self, source, target):
        """String edit distance with equally-weighed features.

        All articulatory features are given equal weight. The distance between
        an unspecified value and a specified value is smaller than the distance
        between two features with oppoiste values.

        Args:
            source (unicode): source string
            target (unicode): target string

        Returns:
            float: feature edit distance with equally-weighed features an insdel
                   costs set so insdel operations cost as much, roughly, as
                   substituting a whole segment
        """
        return self.min_edit_distance(self.unweighted_deletion_cost,
                                      self.unweighted_insertion_cost,
                                      self.unweighted_substitution_cost,
                                      [[]],
                                      self.fm.word_to_vector_list(source),
                                      self.fm.word_to_vector_list(target))

    def jt_feature_edit_distance(self, source, target):
        """String edit distance with equally-weighed features.

        All articulatory features are given equal weight. The distance between
        an unspecified value and a specified value is smaller than the distance
        between two features with oppoiste values. Insdel costs are cheap.

        Args:
            source (unicode): source string
            target (unicode): target string

        Returns:
            float: feature edit distance with equally-weighed features and insdel
                   costs set so insdel operations cost 1/4 as much, roughly, as
                   substituting a whole segment
        """
        return self.min_edit_distance(partial(self.unweighted_deletion_cost, gl_wt=0.25),
                                      partial(self.unweighted_insertion_cost, gl_wt=0.25),
                                      self.unweighted_substitution_cost,
                                      [[]],
                                      self.fm.word_to_vector_list(source),
                                      self.fm.word_to_vector_list(target))


    def feature_edit_distance_div_by_maxlen(self, source, target):
        """Like `Distance.feature_edit_distance` but normalized by maxlen

        Args:
            source (unicode): source string
            target (unicode): target string

        Returns:
            float: feature edit distance with equally-weighed features and insdel
                   costs set so insdel operations cost as much, roughly, as
                   substituting a whole segment

                   Raw result is divided by the length of the longest argument
        """
        maxlen = max(len(source), len(target))
        return self.feature_edit_distance(self, source, target) / maxlen

    def jt_feature_edit_distance_div_by_maxlen(self, source, target):
        """Like `Distance.feature_edit_distance` but normalized by maxlen

        Args:
            source (unicode): source string
            target (unicode): target string

        Returns:
            float: feature edit distance with equally-weighed features and insdel
                   costs set so insdel operations cost 1/4 as much, roughly, as
                   substituting a whole segment

                   Raw result is divided by the length of the longest argument
        """
        maxlen = max(len(source), len(target))
        return self.jt_feature_edit_distance(self, source, target) / maxlen

    def hamming_substitution_cost(self, v1, v2):
        """Substitution cost for feature vectors computed as Hamming distance.

        Substitution cost for feature vectors computed as Hamming distance and
        normalized by dividing this result by the length of the vectors.
        """
        assert len(v1) == len(v2)
        diffs = [ft1 != ft2 for (ft1, ft2) in zip(v1, v2)]
        return sum(diffs) / len(diffs)  # Booleans are cohersed to integers.

    def hamming_feature_edit_distance(self, source, target):
        """String edit distance with equally-weighed features.

        All articulatory features are given equal weight. The distance between an
        unspecified value and a specified value is smaller than the distance between
        two features with oppoiste values.

        The insertion and deletion cost is always one, somewhat favoring
        substitution.

        This function has no normalization but should obey the triangle
        inequality and thus provide a true distance metric.
        """
        return self.min_edit_distance(lambda v: 1,
                                      lambda v: 1,
                                      self.hamming_substitution_cost,
                                      [[]],
                                      self.fm.word_to_vector_list(source),
                                      self.fm.word_to_vector_list(target))

    def hamming_feature_edit_distance_div_maxlen(self, source, target):
        """String edit distance with equally-weighed features divided by maximum length.

        All articulatory features are given equal weight. For substitution, the distance between an
        unspecified value and a specified value is smaller than the distance between
        two features with opposite values. The final value is the string edit
        distance calculated in this way and divided by the length of the longest
        sequence of feature vectors.

        The insertion and deletion cost is always one, somewhat favoring substitution.

        It should be remembered that the resulting function does not obey the
        triangle inequality and is thus not a proper metric.
        """

        source, target = self.fm.word_to_vector_list(source), self.fm.word_to_vector_list(target)
        maxlen = max(len(source), len(target))
        raw = self.min_edit_distance(lambda v: 1,
                                     lambda v: 1,
                                     self.hamming_substitution_cost,
                                     [[]], source, target)
        return raw / maxlen

    def weighted_feature_difference(self, w, ft1, ft2):
        """Return the weighted difference between two features."""
        return w if ft1 != ft2 else 0

    def weighted_substitution_cost(self, v1, v2):
        """Given two feature vectors, return the difference."""
        assert isinstance(v1, list)
        diffs = [self.weighted_feature_difference(w, ft1, ft2)
                 for (w, ft1, ft2) in zip(self.weights, v1, v2)]
        return sum(diffs)

    def weighted_insertion_cost(self, v1):
        """Return cost of inserting segment corresponding to feature vector."""
        assert isinstance(v1, list)
        return sum(self.weights)

    def weighted_deletion_cost(self, v1):
        """Return cost of deleting segment corresponding to feature vector."""
        assert isinstance(v1, list)
        return sum(self.weights)

    def weighted_feature_edit_distance(self, source, target):
        """String edit distance with weighted features.

        The cost of changine an articulatory feature is weighted according to
        the the class of the feature and the subjective probability of the
        feature changing in phonological alternation and loanword contexts.
        """
        return self.min_edit_distance(self.weighted_deletion_cost,
                                      self.weighted_insertion_cost,
                                      self.weighted_substitution_cost,
                                      [[]],
                                      self.fm.word_to_vector_list(source),
                                      self.fm.word_to_vector_list(target))
