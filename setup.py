try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'A Convolutional Neural Network based image processing application for localizing neurons stained with Fluorescent In-Situ Hybridization (FISH)',
    'author': 'Sam Rendall',
    'url': 'https://github.com/sdrendall/fisherman',
    'download_url': 'https://github.com/sdrendall/fisherman.git',
    'author_email': 'sdrendall@gmail.com',
    'version': '0.1',
    'install_requires': ['nose', 'numpy', 'lmdb', 'scikit-image', 'scipy'],
    'packages': ['fisherman'],
    'scripts': [],
    'name': 'fisherman'
}

setup(**config)
