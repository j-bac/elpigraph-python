"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from distutils import util
from os import path
from io import open


here = path.abspath(path.dirname(__file__))
path_src = util.convert_path("elpigraph/src")
# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Get version
ver_path = util.convert_path("elpigraph/_version.py")
main_ns = {}
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)
VERSION = main_ns["__version__"]
# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="elpigraph-python",  # Required
    version=VERSION,  # Required
    description="",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="https://github.com/j-bac/elpigraph-python",  # Optional
    maintainer="Jonathan Bac",  # Optional
    maintainer_email="",  # Optional
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Pick your license as you wish
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    keywords="machine_learning graphs dimension_reduction single_cell",  # Optional
    packages=["elpigraph.src", "elpigraph"],
    package_dir={
        "elpigraph": "elpigraph",
        "elpigraph.src": path_src,
    },
    #     package_data={'': ['data/']},
    install_requires=[
        "numpy >=1.16.2",
        "pandas >=0.23.4",
        "numba >=0.49.1 ",
        "scikit-learn >=0.21.3",
        "scipy >=1.2.0",
        "python_igraph >=0.7.1",
        "networkx >=2.0",
        "matplotlib",
        "shapely",
    ],
    project_urls={  # Optional
        "Bug Reports": "https://github.com/j-bac/elpigraph-python/issues",
        "Source": "https://github.com/j-bac/elpigraph-python/",
    },
    zip_safe=False,
    extras_require={
        "tests": ["pytest", "pytest-cov"],
        "docs": [
            "sphinx",
            "sphinx-gallery",
            "sphinx_rtd_theme",
            "numpydoc",
            "matplotlib",
        ],
    },
)
