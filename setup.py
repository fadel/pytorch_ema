from setuptools import setup, find_packages

__version__ = '0.2'
url = 'https://github.com/fadel/pytorch_ema'
download_url = '{}/archive/{}.tar.gz'.format(url, __version__)

install_requires = []
setup_requires = []
tests_require = []

setup(
    name='torch_ema',
    version=__version__,
    description='PyTorch library for computing moving averages of model parameters.',
    author='Samuel G. Fadel',
    author_email='samuelfadel@gmail.com',
    url=url,
    download_url=download_url,
    keywords=['pytorch', 'parameters', 'deep-learning'],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=find_packages(),
)
