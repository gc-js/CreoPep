from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="creopep",
    version="0.1",
    package_dir={"": "creopep"},
    packages=find_packages(where="creopep"),
    install_requires=requirements,
    python_requires='>=3.10',
    setup_requires=['wheel'],
)
