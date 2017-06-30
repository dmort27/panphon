from __future__ import absolute_import, print_function, unicode_literals

import unicodecsv as csv
import os.path
import pkg_resources


class XSampa(object):
    def __init__(self, delimiter=' '):
        self.delimiter = delimiter
        self.xs2ipa = self.read_xsampa_table()

    def read_xsampa_table(self):
        filename = os.path.join('data', 'ipa-xsampa.csv')
        filename = pkg_resources.resource_filename(__name__, filename)
        with open(filename, 'rb') as f:
            xs2ipa = {x[1]: x[0] for x in csv.reader(f, encoding='utf-8')}
        return xs2ipa

    def convert(self, xsampa):
        def seg2ipa(seg):
            try:
                return self.xs2ipa[seg]
            except KeyError:
                return seg
        return ''.join(map(seg2ipa, xsampa.split(self.delimiter)))
