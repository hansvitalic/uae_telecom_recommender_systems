from setuptools import setup, find_packages

setup(
    name="uae-telecom-recommender",
    version="1.0.0",
    description="Sector-aware recommender systems for project risk management in UAE telecom infrastructure",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Bet_Hans",
    author_email="",
    url="https://github.com/hansvitalic/uae_telecom_recommender_systems",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "pyyaml>=5.4.0",
        "jsonschema>=3.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)