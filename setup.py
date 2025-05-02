from setuptools import setup, find_packages

def read_requirements(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    # Remove comments and empty lines
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]

setup(
    name="probabilistic-methods",
    version="0.1",
    packages=find_packages(),
    package_dir={"": "."},  # Look for packages in the root directory
    install_requires=read_requirements("requirements.txt"),
) 