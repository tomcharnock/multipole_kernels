import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="multipole_kernels",
    version="0.2a",
    author="Tom Charnock",
    author_email="charnock@iap.fr",
    description="TensorFlow convolutional kernels expanded in multipoles to \
        reduce number of weights in kernels",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tomcharnock/multipole_kernels.git",
    packages=["multipole_kernels"],
    py_modules=["multipole_kernels", "multipole_kernels_keras"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
          #"tensorflow>=2.0.0",
          "tqdm>=4.29.0",
          "numpy>=1.16.0",
          "scipy>=1.2.0"
      ],
)
