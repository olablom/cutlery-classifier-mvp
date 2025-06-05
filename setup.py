from setuptools import setup, find_packages

setup(
    name="cutlery-classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "pillow",
    ],
)
