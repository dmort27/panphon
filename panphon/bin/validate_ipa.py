#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TextIO
import panphon
import regex as re
import sys


class Validator(object):
    def __init__(self, infile: TextIO = sys.stdin) -> None:
        """Validate Unicode IPA from file relative to panphon database.

        Parameters
        ----------
        infile : TextIO, optional
            File from which input is taken. Default is sys.stdin.
        """
        self.ws_punc_regex = re.compile(r'[," \t\n]', re.V1 | re.U)
        self.ft = panphon.FeatureTable()
        self._validate_file(infile)

    def _validate_file(self, infile: TextIO) -> None:
        for line in infile:
            self.validate_line(line)

    def validate_line(self, line: str) -> None:
        """Validate Unicode IPA string relative to panphon.

        Parameters
        ----------
        line : str
            String of IPA characters. Can contain whitespace and limited
            punctuation.
        """
        line0 = line
        pos = 0
        while line:
            seg_m = self.ft.seg_regex.match(line)
            wsp_m = self.ws_punc_regex.match(line)
            if seg_m:
                length = len(seg_m.group(0))
                line = line[length:]
                pos += length
            elif wsp_m:
                length = len(wsp_m.group(0))
                line = line[length:]
                pos += length
            else:
                msg = 'IPA not valid at position {} in "{}".'.format(pos, line0.strip())
                print(msg, file=sys.stderr)
                line = line[1:]
                pos += 1


def main():
    """Entry point for the validate_ipa script."""
    Validator(sys.stdin)


if __name__ == '__main__':
    main()
