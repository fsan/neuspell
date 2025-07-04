from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

requirements = [
    'transformers',
    'tqdm',
    'torch>=1.6.0',
    'numpy',
    'jsonlines',
    'sentencepiece',
]

setup(
    name="neuspell",
    version="1.0.0",
    author="Sai Muralidhar Jayanthi, Danish Pruthi, and Graham Neubig",
    author_email="jsaimurali001@gmail.com",
    description="NeuSpell: A Neural Spelling Correction Toolkit",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/neuspell/neuspell",
    packages=find_packages(),
    classifiers=[
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "spacy": ["spacy"],
        "elmo": ["allennlp==1.5.0"],
        "noising": ["unidecode"],
        "flask": ["flask_cors"]
    },
    keywords="transformer networks neuspell neural spelling correction embedding PyTorch NLP deep learning"
)
