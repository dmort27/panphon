from __future__ import absolute_import, print_function, unicode_literals

import regex as re
import unicodecsv as csv
import os.path
import pkg_resources


class XSampa(object):
    def __init__(self, delimiter=' '):
        self.delimiter = delimiter
        self.xs_regex, self.xs2ipa = self.read_xsampa_table()

    def read_xsampa_table(self):
        filename = os.path.join('data', 'ipa-xsampa.csv')
        filename = pkg_resources.resource_filename(__name__, filename)
        with open(filename, 'rb') as f:
            xs2ipa = {x[1]: x[0] for x in csv.reader(f, encoding='utf-8')}
        xs = sorted(xs2ipa.keys(), key=len, reverse=True)
        xs_regex = re.compile('|'.join(list(map(re.escape, xs))))
        return xs_regex, xs2ipa

    def convert(self, xsampa):
        def seg2ipa(seg):
            ipa = []
            while seg:
                match = self.xs_regex.match(seg)
                if match:
                    ipa.append(self.xs2ipa[match.group(0)])
                    seg = seg[len(match.group(0)):]
                else:
                    seg = seg[1:]
            return ''.join(ipa)
        ipasegs = list(map(seg2ipa, xsampa.split(self.delimiter)))
        return ''.join(ipasegs)
