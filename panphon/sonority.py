from __future__ import print_function, absolute_import, unicode_literals

from . import _panphon
from . import permissive

from ._panphon import FeatureTable, fts


class BoolTree(object):
    """Simple decision tree specialized for sonority classes"""
    def __init__(self, test=None, t_node=None, f_node=None):
        """Construct a BoolTree object

        Args:
            test (bool): test for whether to traverse the true-node or the
                         false-node (`BoolTree.t_node` or `BoolTree.f_node`)
            t_node (BoolTree/Int): node to follow if test is `True`
            f_node (BoolTree/Int): node to follow if test is `False`
        """
        self.test = test
        self.t_node = t_node
        self.f_node = f_node

    def get_value(self):
        if self.test:
            if isinstance(self.t_node, BoolTree):
                return self.t_node.get_value()
            else:
                return self.t_node
        else:
            if isinstance(self.f_node, BoolTree):
                return self.f_node.get_value()
            else:
                return self.f_node


class Sonority(object):
    """Determine the sonority of a segment"""
    def __init__(self, feature_set='spe+', feature_model='strict'):
        """Construct a Sonority object

        Args:
            feature_set (str): features set to be used by `FeatureTable`
            feature_model (str): 'strict' or 'permissive' feature model
        """
        fm = {'strict': _panphon.FeatureTable,
              'permissive': permissive.PermissiveFeatureTable}
        self.fm = fm[feature_model](feature_set=feature_set)

    def sonority_from_fts(self, seg):
        """Given a segment as features, returns the sonority on a scale of 1
           to 9.

        Args:
            seg (list): collection of (value, feature) pairs representing
                        a segment (vowel or consonant)

        Returns:
           int: sonority of `seg` between 1 and 9
        """

        def match(m):
            return self.fm.match(fts(m), seg)

        minusHi = BoolTree(match('-hi'), 9, 8)
        minusNas = BoolTree(match('-nas'), 6, 5)
        plusVoi1 = BoolTree(match('+voi'), 4, 3)
        plusVoi2 = BoolTree(match('+voi'), 2, 1)
        plusCont = BoolTree(match('+cont'), plusVoi1, plusVoi2)
        plusSon = BoolTree(match('+son'), minusNas, plusCont)
        minusCons = BoolTree(match('-cons'), 7, plusSon)
        plusSyl = BoolTree(match('+syl'), minusHi, minusCons)
        return plusSyl.get_value()

    def sonority(self, seg):
        """Given a segment as a Unicode IPA string, returns the sonority on
           a scale of 1 to 9.

        Args:
            seg (unicode): IPA consonant or vowel

        Returns:
           int: sonority of `seg` between 1 and 9
        """
        return self.sonority_from_fts(self.fm.fts(seg))
