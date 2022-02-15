from setuptools import setup, find_packages

setup(name='gym_grid_driving',
      version='0.0.1',
      install_requires=['gym', 'numpy', 'torch'],
      packages=find_packages(),
)
