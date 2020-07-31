import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="occ",
    version="0.0.1",
    author="Dongpin oh",
    author_email="dhehdqls@gmail.com",
    description="One-class classifier models for anomaly detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/odb9402/OneClassClassifier",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)