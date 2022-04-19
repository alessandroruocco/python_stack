from setuptools import setup, find_packages
from pyEPOCH.version import __version__

setup(name='pyEPOCH',
      version=__version__,
      packages=find_packages(),
      entry_points={
          'console_scripts': ['pyEPOCH = pyEPOCH.__main__:main']
      }
)
