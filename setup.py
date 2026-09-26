from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_desc = f.read()

setup(
    name="hydromem",
    version="0.0.1.post1",
    author="Soundarya R",
    author_email="soundaryaramachandra2003@gmail.com",
    description="Hydraulic-Inspired Hierarchical Memory for LLMs with Local-First Privacy",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/Soundarya-2004/Hydromem",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.8",
    install_requires=["cryptography>=41.0.0"],
    extras_require={
        "embed": ["fastembed>=0.2.0"],
        "secure": ["argon2-cffi>=21.0.0"],
        "sqlite": [],
        "all": ["fastembed>=0.2.0", "argon2-cffi>=21.0.0"],
    },
)
