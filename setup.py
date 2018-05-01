from setuptools import setup, find_packages
from os import path
import sys


VERSION = '0.1.0'

here = path.abspath(path.dirname(__file__))

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()


if sys.version_info[0] == 2:
    # Mypy doesn't work with Python 2
    mypy = []
elif sys.version_info[0] == 3:
    mypy = ['mypy==0.560']


setup(
    name='onnx-coreml',
    version=VERSION,
    packages=find_packages(exclude=['contrib', 'docs', 'test', 'example']),
    description='Convert ONNX (Open Neural Network Exchange)'
                'models into Apple CoreML format.',
    long_description=long_description,
    url='https://github.com/onnx/onnx-coreml/',
    author='Oleg Poyaganov',
    author_email='oleg@prisma-ai.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    keywords='onnx coreml machinelearning ml coremltools converter neural',
    install_requires=[
        'click',
        'numpy',
        'coremltools>=0.6.3',
        'onnx>=0.2.1',
        'typing>=3.6.4',
        'typing-extensions>=3.6.2.1',
    ],
    setup_requires=['pytest-runner'],
    tests_require=[
        'pytest',
        'pytest-cov',
        'Pillow'
    ],
    extras_require={
        'mypy': mypy,
    },
    entry_points={
        'console_scripts': [
            'convert-onnx-to-coreml = onnx_coreml.bin.convert:onnx_to_coreml'
        ]
    },
)
