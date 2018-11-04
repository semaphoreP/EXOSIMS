from setuptools import setup, find_packages

setup(
    name='EXOSIMS',
    version='0.1',
    description='EXOSIMS',
    install_requires=['numpy', 'scipy', 'astropy'],
    packages=find_packages(),
    zip_safe=False
    )
