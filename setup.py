from setuptools import setup, find_namespace_packages

setup(
    name="cutlery-classifier",
    version="0.1.0",
    packages=find_namespace_packages(where="src", include=["*"]),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "pillow>=8.0.0",
        "opencv-python>=4.5.0",  # For conveyor simulation
        "tqdm>=4.60.0",  # For progress bars
    ],
    python_requires=">=3.8",
)
