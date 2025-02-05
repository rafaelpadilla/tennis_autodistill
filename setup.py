from setuptools import setup, find_packages

__version__ = '0.1.0'

setup(
    name='tennis_autodistillation',
    version=__version__,
    author='Rafael Padilla',
    author_email='eng.rafaelpadilla@gmail.com',
    description='A project for autodistillation in tennis analytics',
    long_description=open('README.md').read(),
    url='https://github.com/rafaelpadilla/tennis-autodistill',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[],
)
