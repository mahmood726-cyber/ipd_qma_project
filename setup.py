"""
Setup configuration for IPD-QMA PyPI package.

Install with:
    pip install -e .

Build distribution:
    python setup.py sdist bdist_wheel

Upload to PyPI:
    twine upload dist/*
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_file(filename):
    """Read file contents."""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, filename), encoding='utf-8') as f:
        return f.read()

# Get version from __version__ in main module
def get_version():
    """Extract version from ipd_qma.py."""
    with open('ipd_qma.py', 'r') as f:
        for line in f:
            if '__version__' in line:
                # Extract version string
                version = line.split('=')[1].strip().strip('"').strip("'")
                return version
    return '2.0.0'  # Default fallback

setup(
    name='ipd-qma',
    version=get_version(),
    author='IPD-QMA Development Team',
    author_email='your-email@example.com',
    description='Individual Participant Data Quantile Meta-Analysis for detecting heterogeneous treatment effects',
    long_description=read_file('README.md') if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/ipd-qma',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/ipd-qma/issues',
        'Source': 'https://github.com/yourusername/ipd-qma',
        'Documentation': 'https://ipd-qma.readthedocs.io/',
    },
    packages=find_packages(exclude=['tests', 'tests.*', 'benchmarks', 'benchmarks.*', 'docs', 'examples']),
    classifiers=[
        # Development Status
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',

        # Python Version
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',

        # License
        'License :: OSI Approved :: MIT License',

        # Operating System
        'Operating System :: OS Independent',
    ],
    keywords='meta-analysis ipd quantile heterogeneous-effects treatment-effects bootstrap statistics',
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'matplotlib>=3.3.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
            'black>=22.0',
            'flake8>=5.0',
            'mypy>=1.0',
        ],
        'plots': [
            'plotly>=5.0',
        ],
        'progress': [
            'tqdm>=4.60',
        ],
        'export': [
            'openpyxl>=3.0',
        ],
        'web': [
            'streamlit>=1.20',
        ],
        'all': [
            'plotly>=5.0',
            'tqdm>=4.60',
            'openpyxl>=3.0',
            'streamlit>=1.20',
        ],
    },
    entry_points={
        'console_scripts': [
            'ipd-qma=ipd_qma:run_tutorial',
            'ipd-qma-benchmark=benchmarks.benchmark_ipd_qma:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
