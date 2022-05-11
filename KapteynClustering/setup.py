from setuptools import setup, find_packages
packages = find_packages(include=['KapteynClustering'])
#include=['*.py'])#
print('My Packages:')
print(packages)
setup( name='KapteynClustering',
      packages =packages
     )
