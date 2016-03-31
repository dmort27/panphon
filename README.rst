panphon
=======

These files constitute a preliminary database of segments in the
International Phonetic Alphabet (IPA) and their equivalents in terms of
(articulatory) phonological features. They include both data files and
the tool ``apply_diacritics.py``, which allows the application of rules
for diacritics and modifiers to collections of IPA characters, data
files, and configuration/rule files and well as the tool
``validate_ipa.py``, which checks Unicode IPA text from STDIN for
well-formedness.

Installation
============

``panphon`` has a few library dependencies aside from the Python
standard library:

1. `PyYAML <http://pyyaml.org/wiki/PyYAML>`__. If ``pip`` is installed,
   you can easily install PyYAML:

   $ pip install PyYAML

2. `unicodecsv <https://pypi.python.org/pypi/unicodecsv/0.9.4>`__. It is
   likewise trivial to install ``unicodecsv`` if ``pip`` is installed:

   $ pip install unicodecsv

3. `regex <https://pypi.python.org/pypi/regex>`__. The installation
   procedure for installing ``regex`` is the same as that for ``PyYAML``
   and ``unicodecsv`` if ``pip`` is installed:

   $ pip install regex

To install ``panphon``, uncompress the source distribution, change
directories to the root directory of the source distribution, and run
``python setup.py install``:

::

    $ tar xzf panphon-0.1.tar.gz
    $ cd panphon-0.1
    $ python setup.py install

Of, if appropriate for your system:

::

    $ sudo python setup.py install

Python API for Accessing Phonological Features of IPA Segments
--------------------------------------------------------------

The ``panphon`` module provides a straightforward API that allows users
and developers to access the segment-feature relationships encoded in
the IPA database ``panphon/data/segment_features.csv``.

::

    >>> import panphon.panphon as panphon
    >>> ft = panphon.FeatureTable()
    >>> ft.ftr_match(set([(u'+', u'syl')]), u'a')
    True

The interface of ``FeatureTable`` has grown to be quite complicated. Not
all of the methods will be illustrated or documented here. However, all
significant methods of ``FeatureTable`` have detailed docstrings.

Operations on feature sets and segments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``FeatureTable`` class includes a broad range of operations on
features and segments.

Fixed-width pattern matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``FeaturTable`` class allows matching of Fixed-width, feature-based
patterns.

Sonority calculations
~~~~~~~~~~~~~~~~~~~~~

The ``FeatureTable`` class has methods for computing sonority scores for
segments.

Feature edit distance
~~~~~~~~~~~~~~~~~~~~~

The ``FeatureTable`` class includes methods for calculating edit
distance, both in which the cost of substitutions is based upon Hamming
distance between the feature vectors and in which the cost of
substitutions are based upon edit weights for individual features.

Diacritic Application Tool: apply\_diacritics.py
------------------------------------------------

This small, self-documenting Python program allows the user to apply
sets of rules, defined in YAML, for adding diacritics and modifiers to
IPA segments based upon their phonological features. For a detailed help
message, execute ``python apply_diacritics.py -h`` in the directory
containing ``apply_diacritics.py``.

Output Files
~~~~~~~~~~~~

The tool ``apply_diacritics.py`` produces three kinds of output,
depending on the command line options that are specified:

1. *Segments file.* This is a CSV file containing a header row with the
   names of features and a succession of non-header rows, each of which
   includes an IPA segment in the first field and feature specifications
   in each of the subsequent fields. When diacritics are applied to
   bases to produce new segments, these are added to this list. The
   sorting of this list is controlled by the **sort order
   specification**. By default, this file is called
   ``segment_features.csv``. The name can be explicitly specified with
   the ``-o`` or ``--output`` options.
2. *Diacritics/modifier file.* This is a CSV file containing a header
   row with the names of each series. Each column contains a series of
   IPA segments to which a particular diacritic or modifier has been
   applied. These are likewise sorted according to the **sort order
   specification** as applied to the first column.
3. *Segment comparison files.* These two files are produced when the
   ``-c`` or ``--compare`` options are specified. The first contains the
   segments predicted by the model but not attested in the specified
   standards of comparison (e.g. ``phoible_phonemes.csv``); the second
   contains the segments attested in the standard of comparison but not
   predicted by the model associated with the database.

Usage
~~~~~

To generate a segment features file (``segment_features.csv``) and a
segment series file (``segment_series.csv``), use the following:

``python apply_diacritics.py ipa_table.csv diacritic_definitions.yml -a -f segment_features.csv -o segment_series.csv``

This is the most common use of the script. To sort the segments by
feature specification, include a sort order file with the ``-s`` or
``--sort`` option:

``python apply_diacritics.py ipa_table.csv diacritic_definitions.yml -a -s sort_order.yml -f segment_features.csv -o segment_series.csv``

The script can also be used to compare the results of applying all of
the diacritics/modifiers in ``diacritic_definitions.yml`` to the
segments in ``ipa_table.csv`` to other lists of phonemes (e.g. from the
PHOIBLE database). This is done with the ``-c`` or ``--compare`` option,
which takes three arguments:

1. The file to which the comparison is to be made.
2. The file to which the segments which are predicted by the model by
   not attested in the list are to be written.
3. The file to which the segments which are attested in the list but not
   predicted by the model are to be written.

Data Files
----------

This package also includes one data file. The most important of these is
ipa\_table.csv, a CSV table of IPA characters with definitions in terms
of phonological features.

IPA Character Database: ipa\_table.csv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

The data in this table is drawn primarily from two sources: the data
files for `HsSPE <https://github.com/dmort27/HsSPE>`__ and Bruce Hayes's
`feature
spreadsheet <http://www.linguistics.ucla.edu/people/hayes/IP/#features>`__.

The IPA Character Table is intended to contain all of the unmodified
symbols in IPA, as well as all common affricates and dually-articulated
segments. It is meant to be augmented by the rule-driven application of
diacritics and modifiers.

Configuration and Rule Files
----------------------------

This package includes two files that control the behavior of the
Diacritic Application Tool. These are intended to be edited by the end
user. Both are written in `YAML <http://www.yaml.org/>`__, a
human-readable and editable data serialization standard.

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
