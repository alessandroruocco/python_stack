from setuptools import setup, find_packages

setup(name='sdfUtils',
      packages=find_packages(),
      entry_points={
          'console_scripts': ['sdfUtils = sdfUtils.__main__:main',
                              'lsPrefix = sdfUtils.lsPrefix:main']
      }
)
