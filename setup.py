from setuptools import setup

setup(name='panphon',
      version='0.1',
      description='Tools for using the International Phonetic' +
      'Alphabet with phonological features',
      url='http://www.davidmortensen.org/panphon',
      author='David R. Mortensen',
      author_email='dmortens@cs.cmu.edu',
      license='MIT',
      install_requires=['setuptools',
                        'unicodecsv',
                        'PyYAML',
                        'regex'],
      scripts=['panphon/bin/apply_diacritics.py',
               'panphon/bin/validate_ipa.py'],
      packages=['panphon'],
      package_dir={'panphon': 'panphon'},
      package_data={'panphon': ['data/*.csv', 'data/*.yml']},
      zip_safe=True
      )
