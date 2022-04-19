from setuptools import setup, find_packages

setup(name='pbsUtils',
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'pbsUtils = pbsUtils.__main__:main'
          ]
      }
)
