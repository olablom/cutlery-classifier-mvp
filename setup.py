from setuptools import setup, find_packages

setup(
    name="cutlery-classifier",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Core ML
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pillow>=9.0.0",
        "scikit-learn>=1.0.0",
        # Computer Vision
        "opencv-python>=4.7.0",
        "grad-cam>=1.4.0",
        # Visualization
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        # Model Export
        "onnx>=1.13.0",
        "onnxruntime>=1.13.0",
        # Training & Evaluation
        "optuna>=3.0.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        # Utilities
        "PyYAML>=6.0.0",
        "coloredlogs>=15.0.0",
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-grade cutlery classifier with explainability",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cutlery-classifier-mvp",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
