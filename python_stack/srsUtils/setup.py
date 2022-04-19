from setuptools import setup, find_packages

setup(name='srsUtils',
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'srsUtils = srsUtils.__main__:main'
          ]
      }
)
