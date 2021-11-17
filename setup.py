from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / "torch_ema/_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

url = 'https://github.com/fadel/pytorch_ema'
download_url = '{}/archive/{}.tar.gz'.format(url, version)

install_requires = ["torch"]
setup_requires = []
tests_require = []

setup(
    name='torch_ema',
    version=version,
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
