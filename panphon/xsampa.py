
import regex as re
import csv
import os.path
from importlib.resources import files


class XSampa(object):
    def __init__(self, delimiter=' '):
        self.delimiter = delimiter
        self.xs_regex, self.xs2ipa = self.read_xsampa_table()

    def read_xsampa_table(self):
        filename = os.path.join('data', 'ipa-xsampa.csv')
        filename = files('panphon').joinpath(filename)
        with filename.open('r', encoding='utf-8') as f:
            xs2ipa = {x[1]: x[0] for x in csv.reader(f)}
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
