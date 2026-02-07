from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cutlery-classifier",
    version="3.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "numpy>=1.21.0",
        "pillow>=8.3.0",
        "onnx>=1.12.0",
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
    description="Offline inference MVP for cutlery classification",
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
