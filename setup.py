import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ARMxtend",
    version="0.0.1",
    author="Carlos Fernandez-Basso",
    author_email="cjferba@decsai.ugr.es",
    description="ARMxtend (association rule mining extensions)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cjferba/ARMxtend",
    license="CC-BY-NC-4.0",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "networkx",
    ],
    extras_require={
        # Necesario solo para los algoritmos de Big Data/streaming:
        # FIM.apriori, FIM.Eclat, FIM.BD_ARE, FIM.BD_FARE, SARE, SFIM
        "spark": ["pyspark"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)