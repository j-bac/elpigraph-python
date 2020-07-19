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

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="elpigraph-python",  # Required
    version="0.1.0",  # Required
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
    package_dir={"elpigraph": "elpigraph", "elpigraph.src": path_src,},
    #     package_data={'': ['data/']},
    install_requires=["numpy", "pandas", "scipy", "scikit_learn", "python_igraph"],
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
