from setuptools import setup

setup(name='panphon',
      version='0.2',
      description='Tools for using the International Phonetic Alphabet with phonological features',
      url='https://github.com/dmort27/panphon',
      download_url='https://github.com/dmort27/panphon/tarball/0.2',
      author='David R. Mortensen',
      author_email='dmortens@cs.cmu.edu',
      license='MIT',
      install_requires=['setuptools',
                        'unicodecsv',
                        'PyYAML',
                        'regex',
                        'numpy'],
      scripts=['panphon/bin/apply_diacritics.py',
               'panphon/bin/validate_ipa.py',
               'panphon/bin/align_wordlists.py'],
      packages=['panphon'],
      package_dir={'panphon': 'panphon'},
      package_data={'panphon': ['data/*.csv', 'data/*.yml']},
      zip_safe=True
      )
