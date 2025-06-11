from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cutlery-classifier",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Core ML
        "torch==2.3.1",
        "torchvision==0.18.1",
        "numpy==1.26.4",
        "pillow==11.2.1",
        "scikit-learn==1.6.1",
        # Computer Vision
        "opencv-python==4.11.0.86",
        "grad-cam==1.5.5",
        # Visualization
        "matplotlib==3.10.3",
        "seaborn==0.13.2",
        # Model Export
        "onnx==1.18.0",
        "onnxruntime==1.22.0",
        # Training & Evaluation
        "optuna==3.5.0",
        "tqdm==4.67.1",
        "pandas==2.2.3",
        # Utilities
        "PyYAML==6.0.2",
        "coloredlogs==15.0.1",
        # Testing
        "pytest==8.4.0",
        "pytest-cov==6.1.1",
    ],
    entry_points={
        "console_scripts": [
            "cutlery-inference=cutlery_classifier.scripts.run_inference:main",
            "cutlery-test=cutlery_classifier.scripts.test_dataset_inference:main",
            "cutlery-train=cutlery_classifier.scripts.train_type_detector:main",
        ],
    },
    python_requires=">=3.8",
    author="Ola Blom",
    author_email="ola.blom@example.com",
    description="Production-grade cutlery classifier with explainability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/olablom/cutlery-classifier-mvp",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
