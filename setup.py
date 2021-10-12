from setuptools import setup
from setuptools import find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="kaustav-ai-workspace",
    version="1.0",
    description="Collection of ai algorithms",
    author="Kaustav Ghosh",
    author_email="teetangh@gmail.com",

    python_requires='>3.9.0',
    packages=find_packages(
        exclude=("tests*", "testing*", "examples*", "docs*", "build*")
    ),  # finds all folders with __init__.py

    install_requires=required,

    # install_requires=[
    #     "numpy", "pandas", "scikit-learn", "matplotlib",
    #     "tensorflow", "tensorflow-gpu",
    #     "torch", "torchvision", "torchaudio", "keras", "gym"],

    # entry_points={
    #     "console_scripts": [
    #         "cartpole-qlearning-cli=reinforcement-learning.qlearning.main:cartpole"
    #     ]
    # },

)
