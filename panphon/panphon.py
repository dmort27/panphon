# -*- coding: utf-8 -*-

import unicodecsv as csv


class FeatureTable(object):
    def __init__(self):
        self._read_table(self)
        self.seg_dict = dict(self.segments)

    def _read_table(self):
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

    def feature_match(self, features, segment):
        return segment in [s for (s, fts)
                           in self.segments
                           if features <= fts]
