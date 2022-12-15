#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name='phalp',
    version='0.1.2',    
    description='PHALP: A Python package for People Tracking in 3D',
    url='https://github.com/brjathu/PHALP',
    author='Jathushan Rajasegaran',
    author_email='jathushan@berkeley.edu',
    license='MIT License',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        "hydra-core",
        # "opencv-python",
        # "joblib",
        # "cython",
        # "scikit-learn==0.22",
        # "scikit-image",
        # "chumpy",
        # "ipython",
        # "gdown",
        # "dill",
        # "rich",
        # "python-dotenv",
        # "motmetrics",
        # "pyrootutils",
        # "scenedetect[opencv]",
        # "pyopengl==3.1.6",
        # "pyrender==0.1.45",
        # "networkx==2.8.6",
        ],
)
