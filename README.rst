PanPhon
=======

This package constitutes a database of segments in the International
Phonetic Alphabet (IPA) and their equivalents in terms of (articulatory)
phonological features. They include both data files and the tool
``generate_ipa_all.py``, which allows the application of rules for
diacritics and modifiers to collections of IPA characters, data files,
and configuration/rule files and well as the tool ``validate_ipa.py``,
which checks Unicode IPA text from STDIN for well-formedness.

Python API for Accessing Phonological Features of IPA Segments
--------------------------------------------------------------

The ``panphon`` module provides a straightforward API that allows users
and developers to access the segment-feature relationships encoded in
the IPA database ``panphon/data/ipa_all.csv``.

.. code:: python

    >>> import panphon.panphon as panphon
    >>> ft = panphon.FeatureTable()
    >>> ft.ftr_match(set([(u'+', u'syl')]), u'a')
    True
    >>> ft.segs(u'pʲãk')
    [u'p\u02b2', u'a\u0303', u'k']
    >>> ft.word_fts(u'pʲãk')
    [set([(u'-', u'syl'), (u'-', u'long'), (u'-', u'voi'), (u'+', u'ant'), (u'-', u'cg'), (u'+', u'hi'), (u'-', u'son'), (u'0', u'tense'), (u'-', u'lat'), (u'-', u'back'), (u'-', u'cont'), (u'-', u'nas'), (u'-', u'lo'), (u'0', u'distr'), (u'-', u'round'), (u'-', u'delrel'), (u'+', u'lab'), (u'-', u'sg'), (u'+', u'cons'), (u'0', u'strid'), (u'-', u'cor')]), set([(u'+', u'son'), (u'+', u'tense'), (u'+', u'cont'), (u'+', u'nas'), (u'+', u'lo'), (u'+', u'voi'), (u'-', u'cg'), (u'-', u'hi'), (u'-', u'lat'), (u'+', u'syl'), (u'0', u'strid'), (u'-', u'long'), (u'-', u'cor'), (u'0', u'distr'), (u'-', u'round'), (u'-', u'delrel'), (u'0', u'ant'), (u'-', u'sg'), (u'+', u'back'), (u'-', u'cons'), (u'-', u'lab')]), set([(u'-', u'syl'), (u'-', u'lab'), (u'-', u'voi'), (u'0', u'distr'), (u'+', u'back'), (u'-', u'cg'), (u'+', u'hi'), (u'-', u'son'), (u'0', u'tense'), (u'-', u'lat'), (u'-', u'cont'), (u'-', u'nas'), (u'-', u'lo'), (u'-', u'ant'), (u'-', u'round'), (u'-', u'delrel'), (u'-', u'sg'), (u'+', u'cons'), (u'0', u'strid'), (u'-', u'cor'), (u'-', u'long')])]

Summary of Functionality
------------------------

Operations on feature sets and segments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``FeatureTable`` class includes a broad range of operations on
features and segments (consonants and vowels).

Converting words to feature arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``panphon`` class includes the function word2array which takes a
list of feature names (as a list of strings) and a panphon word (from
FeatureTable().word\_fts()) and returns a NumPy array where each row
corresponds to a segment in the word and each column corresponds to one
of the specified features. Basic usage is illustrated in the following
example:

.. code:: python

    >>> import panphon
    >>> ft=panphon.FeatureTable()
    >>> panphon.word2array(['syl', 'son', 'cont'], ft.word_fts(u'snik'))
    array([[-1, -1,  1],
           [-1,  1, -1],
           [ 1,  1,  1],
           [-1, -1, -1]])

Fixed-width pattern matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``FeatureTable`` class also allows matching of fixed-width,
feature-based patterns.

Sonority calculations
~~~~~~~~~~~~~~~~~~~~~

The ``Sonority`` class has methods for computing sonority scores for
segments.

Feature edit distance
~~~~~~~~~~~~~~~~~~~~~

The ``Distance`` class includes methods for calculating edit distance,
both in which the cost of substitutions is based upon Hamming distance
between the feature vectors and in which the cost of substitutions are
based upon edit weights for individual features.

The ``panphon.distance`` Module
-------------------------------

This module includes the ``Distance`` class, which includes various
methods for computing the distance between Unicode IPA strings,
including convenience methods (really "inconvenience methods") for
computing Levenshtein distance, but--more importantly--methods for
computing similarity metrics related to articulatory features. The
methods include the following:

``panphon.distance.Distance`` .\ **levenshtein\_distance**

A Python implementation of Levenshtein's string edit distance.

``panphon.distance.Distance`` .\ **fast\_levenshtein\_distance**

A C implementation of Levenshtein's string edit distance.
Unsurprisingly, must faster than the former.

``panphon.distance.Distance`` .\ **dogol\_prime\_distance**

Fast Levenshtein distance after collapsing segments into an enhanced
version of Dogolpolsky's equivalence classes.

``panphon.distance.Distance`` .\ **feature\_edit\_distance**

Edit distance where each feature-edit has cost 1/22. Edits from
unspecified to specified cost 1/44.

``panphon.distance.Distance`` .\ **hamming\_feature\_edit\_distance**

Edit distance where each feature-edit has cost 1/22. Edits from
unspecified to specified also cost 1/22. Insertions and substitutions
each cost 1.

``panphon.distance.Distance`` .\ **weighted\_feature\_edit\_distance**

Edit distance where costs of feature edits are differently weighted
depending on their class and subjective variability. All of these
methods have the same interface and patterns of usage, demonstrated
below:

.. code:: python

    >>> import panphon.distance
    >>> dst = panphon.distance.Distance()
    >>> dst.dogol_prime_distance(u'pops', u'bobz')
    0
    >>> dst.dogol_prime_distance(u'pops', u'bobo')
    1

Scripts
-------

The ``generate_ipa_all.py`` Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Summary
^^^^^^^

This small Python program allows the user to apply sets of rules,
defined in YAML, for adding diacritics and modifiers to IPA segments
based upon their phonological features.

Usage
^^^^^

To generate a segment features file (``ipa_all.csv``), use the following
**in the panphon data directory**:

.. code:: bash

    $ generate_ipa_all.py ipa_bases.csv -d diacritic_definitions.yml -s sort_order.yml ipa_all.csv

Note that this will overwrite your existing ``ipa_all.csv`` file, which
is often what you want.

The ``validate_ipa.py`` Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

[To be added.]

The ``align_wordlists.py`` Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

[To be added.]

Data Files
----------

This package also includes multiple data files. The most important of
these is ipa\_bases.csv, a CSV table of IPA characters with definitions
in terms of phonological features. From it, and the
``diacritics_definitions.yml`` file, the comprehensive ``ipa_all.csv``
is generated.

IPA Character Databases: ``ipa_bases.csv`` and ``ipa_all.csv``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The IPA Character Table is a CSV file in which the first column contains
an IPA segment and each subsequent column contains a phonological
feature, coded as +, -, or 0. The features are as follows:

-  **syl**: syllabic
-  **son**: sonorant
-  **cons**: consonantal
-  **cont**: continuant
-  **delrel**: delayed release
-  **lat**: lateral
-  **nas**: nasal
-  **strid**: strident
-  **voi**: voice
-  **sg**: spread glottis
-  **cg**: constricted glottis
-  **ant**: anterior
-  **cor**: coronal
-  **distr**: distributed
-  **lab**: labial
-  **hi**: high (vowel/consonant, not tone)
-  **lo**: low (vowel/consonant, not tone)
-  **back**: back
-  **round**: round
-  **tense**: tense

Inspiration for the data in these tables is drawn primarily from two
sources: the data files for `HsSPE <https://github.com/dmort27/HsSPE>`__
and Bruce Hayes's `feature
spreadsheet <http://www.linguistics.ucla.edu/people/hayes/IP/#features>`__.
It has since be re-rationalizeds based on evidence from a wide range of
sources. As such, any special relationship to these prior inspirations
has been eliminated.

The IPA Character Table ``ipa_bases.csv`` is intended to contain all of
the unmodified segmental symbols in IPA, as well as all common
affricates and dually-articulated segments. It is meant to be augmented
by the rule-driven application of diacritics and modifiers.

Configuration and Rule Files
----------------------------

This package includes two files that control the behavior of
``generate_ipa_all.py``. These are intended to be edited by the end
user. Both are written in `YAML <http://www.yaml.org/>`__, a
standardized and human-readable and editable data serialization
language.

Sort Order Specification: sort\_order.yml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The file ``sort_order.yml`` controls the ordering of segments in the
output of the Diacritic Application Tool. It is a sequence of maps, each
with two fields:

-  **name** The name of a feature.
-  **reverse** A boolean value (True or False) specifying whether
   sorting on the named feature will be reversed or not.

The order of the features determines the priority of sorting.

The file ``sort_order_schema_.yml`` is a
`Kwalify <http://www.kuwata-lab.com/kwalify/>`__ schema that defines a
syntactically valid sort order file.

Diacritic and Modifier Rules: diacritic\_definitions.yml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most important file for controlling the Diacritic Application Tool
is ``diacritic_definitions.yml``, a list of rules for applying
diacritics and modifiers to IPA segments based on their phonological
features. It has two sections, **diacritics** and **combinations**. Each
of these is the key to an item in the top-level map.

Diacritics
^^^^^^^^^^

The key **diacritics** points to a list of rules for applying
diacritics/modifiers to bases. Each rule is a map with the following
fields:

-  **marker.** The Unicode diacritic or modifier.
-  **name.** The name of the series derived from applying the diacritic
   or modifier.
-  **postion.** The position of the diacritic relative to the base (pre
   or post).
-  **conditions.** A list of conditions, each of them consisting of an
   associative array of feature specifications, under which the
   diacritic or modifier will be applied to a base.
-  **exclude.** A sequence of segments to be excluded from the
   application of the diacritic/modifier even if they match the
   conditions.
-  **content.** The feature specifications that will be set if the
   diacritic or modifier is applied, given as a map of feature
   specifications.

Combinations
^^^^^^^^^^^^

The key **combinations** likewise points to a list of rules for
combining the rules in **diacritics**. These rules are very simple, and
include only the following fields:

-  **name.** The name of the combined category.
-  **combines.** A sequence of the names of the rules from
   **diacritics** that are to be combined.

The file ``diacritic_definitions_schema.yml`` is a
`Kwalify <http://www.kuwata-lab.com/kwalify/>`__ schema that defines a
syntactically valid diacritics definition file.

Citing PanPhon
==============

If you use PanPhon in research, please cite the following paper:

David R. Mortensen, Patrick Littell, Akash Bharadwaj, Kartik Goyal,
Chris Dyer, Lori Levin (2016). "PanPhon: A Resource for Mapping IPA
Segments to Articulatory Feature Vectors." *Proceedings of COLING 2016,
the 26th International Conference on Computational Linguistics:
Technical Papers*, pages 3475–3484, Osaka, Japan, December 11-17 2016.
