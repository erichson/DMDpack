# Install setuptools if it isn't available:
try:
    import setuptools
except ImportError:
    print "The package 'setuptools' is required!"

NAME ='DMDpack'
VERSION ='1.0.0'
DESCRIPTION ='Dynamic Mode Decomposition'
URL ='https://github.com/Benli11/DMDpack'
AUTHER ='N. Benjamin Erichson'
EMAIL ='nbe@st-andrews.ac.uk'
LICENSE ='BSD'
#PACKAGES = ['dmd', 'tools']

      
      
try:
    from setuptools import setup, find_packages
except ImportError:
    print "The package 'setuptools' is required!"

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name = NAME,
    version = VERSION,
    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,
    url = URL,
    author = AUTHER,
    author_email = EMAIL,
    license = LICENSE,

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Researchers',
        'Topic :: Randomized Linear Algebra :: Video Processing',

        # Pick your license as you wish (should match "license" above)
        'License :: BSD',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        #'Programming Language :: Python :: 2',
        #'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        #'Programming Language :: Python :: 3',
        #'Programming Language :: Python :: 3.2',
        #'Programming Language :: Python :: 3.3',
        #'Programming Language :: Python :: 3.4',
    ],

    # What does your project relate to?
    keywords='dynamic mode decomposition, randomized singular value decomposition, object detection, rsvd, rpca, dmd',

    #packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    packages=find_packages(exclude=['tests*']),
    test_suite='nose.collector'	
)