from setuptools import setup, find_packages
packages = find_packages(include=['KapteynClustering'])
print('My Packages:')
print(packages)
setup( name='KapteynClustering',
      packages =packages,
      author_email="t.m.callingham@astro.rug.nl",
     )
