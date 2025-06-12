from setuptools import setup

setup(name='panphon',
      version='0.22.2',
      description='Tools for using the International Phonetic Alphabet ' +
      'with phonological features',
      url='https://github.com/dmort27/panphon',
      download_url='https://github.com/dmort27/panphon/archive/0.21.1.tar.gz',
      long_description=open('README.md', encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      author='David R. Mortensen',
      author_email='dmortens@cs.cmu.edu',
      license='MIT',
      install_requires=['setuptools',
                        'unicodecsv',
                        'PyYAML',
                        'regex',
                        'numpy>=1.20.2',
                        'editdistance',
                        'munkres',
                        'pandas',],
      scripts=['panphon/bin/validate_ipa.py',
               'panphon/bin/align_wordlists.py',
               'panphon/bin/generate_ipa_all.py'],
      packages=['panphon'],
      package_dir={'panphon': 'panphon'},
      package_data={
            'panphon': [
                  'data/*.csv',
                  'data/*.yml',
            ]
      },
      zip_safe=True,
      classifiers=[
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: Text Processing :: Linguistic']
      )
