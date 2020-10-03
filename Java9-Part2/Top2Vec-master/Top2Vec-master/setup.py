import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="top2vec",
    packages=["top2vec"],
    version="1.0.10",
    author="Dimo Angelov",
    author_email="dimo.angelov@gmail.com",
    description="Top2Vec learns jointly embedded topic, document and word vectors.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ddangelov/Top2Vec",
    keywords="topic modeling semantic search word document embedding",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas',
        'gensim',
        'pynndescent >= 0.4',
        'umap-learn',
        'hdbscan',
        'wordcloud',
      ],
    python_requires='>=3.6',
)
