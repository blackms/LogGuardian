from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="logguardian",
    version="0.1.0",
    author="Alessio Rocchi",
    author_email="rocchi.b.a@gmail.com",
    description="A Python-based log anomaly detection system leveraging LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/logguardian",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "loguru>=0.7.0",
        "pandas>=1.5.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
    },
)