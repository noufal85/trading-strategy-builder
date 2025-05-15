from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="strategy_builder",
    version="0.1.0",
    author="Trading Strategy Builder Team",
    author_email="example@example.com",
    description="A framework for building and backtesting trading strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/noufal85/trading-strategy-builder",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
