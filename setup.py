"""Offline-compatible setuptools entry point; canonical metadata lives in pyproject.toml."""

from setuptools import find_packages, setup

setup(
    name="evofactskill",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={"console_scripts": ["evofact=evofact.cli:main"]},
    python_requires=">=3.11",
)
