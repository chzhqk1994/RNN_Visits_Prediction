'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='rnn_model',
      version='1.4',
      packages=find_packages(),
      include_package_data=True,
      description='Page_impressions keras model on Cloud ML Engine',
      author='Your Name',
      author_email='hyunwoo.song@mamamia.com.au',
      license='MIT',
      install_requires=[
          'keras',
          'h5py'],
      zip_safe=False)