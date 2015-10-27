from setuptools import setup

setup(name='panphon',
      version='0.1',
      description='Tools for using the International Phonetic' +
      'Alphabet with phonological features',
      url='http://www.davidmortensen.org/panphon',
      author='David R. Mortensen',
      author_email='dmortens@cs.cmu.edu',
      license='MIT',
      packages=['panphon'],
      install_requires=['unicodecsv',
                        'PyYAML',
                        'regex'],
      scripts=['panphon/bin/apply_diacritics.py'],
      package_data={'': ['data/*.csv',
                         'data/*.yml']},
      zip_safe=True
      )
