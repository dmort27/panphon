# -*- coding: utf-8 -*-

import unicodecsv as csv


class FeatureTable(object):
    """Encapsulate the segment <=> feature mapping in the file
    data/segment_features.csv.

    """
    def __init__(self):
        self._read_table()

    def _read_table(self):
        """Read the data from data/segment_features.csv into self.segments, a
        list of 2-tuples of unicode strings and sets of feature tuples
        and self.seg_dict, a dictionary mapping from unicode segments
        and sets of feature tuples.

        """
        self.segments = []
        with open('data/segment_features.csv', 'rb') as f:
            reader = csv.reader(f, encoding='utf-8')
            header = reader.next()
            names = header[1:]
            for row in reader:
                seg = row[0]
                vals = row[1:]
                specs = set(zip(vals, names))
                self.segments.append((seg, specs))
        self.seg_dict = dict(self.segments)
        # A few sanity checks:
        assert (u'+', u'cons') in self.seg_dict[u'tʰ']
        assert set[(u'-', u'cons'), (u'-', u'hi')] < self.seg_dict[u'a']

    def feature_match(self, features, segment):
        """Evaluates whether a set of features 'match' a segment (are a subset
        of that segment's features).
        """
        return segment in [s for (s, fts)
                           in self.segments
                           if features <= fts]
