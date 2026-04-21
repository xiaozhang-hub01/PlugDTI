from setuptools import find_packages, setup

setup(
    name="plugdti",
    version="0.1.0",
    description="Plug-and-play pretrained sequence representation plugin for DTI/DTA backbones",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23",
        "torch>=2.0",
        "transformers>=4.35",
    ],
    python_requires=">=3.9",
)
