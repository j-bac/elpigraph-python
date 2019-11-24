"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='elpigraph',  # Required
    version='1.0',  # Required
    description='',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/j-bac/ElPiGraph.Python',  # Optional
    author='Jonathan Bac',  # Optional
    author_email='',  # Optional
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Pick your license as you wish
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'],

    keywords='machine_learning graphs dimension_reduction single_cell',  # Optional
    packages=['elpigraph'],
#     package_dir={'elpigraph': 'elpigraph'},
    package_data={'elpigraph': ['data/*.csv']},
    install_requires=['numpy','pandas','scipy','scikit_learn','python_igraph','plotnine'],
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/j-bac/ElPiGraph.Python/issues',
        'Source': 'https://github.com/j-bac/ElPiGraph.Python/',
    },
)
