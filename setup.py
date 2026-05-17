from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="modelshift",
    version="0.2.0",  # Update this every time you publish a new version
    author="Krishna",
    author_email="ryomensukuna2530@gmail.com", 
    description="A lightweight machine learning drift monitoring and alerting engine.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["modelshift", "modelshift.*"]),
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "requests"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)