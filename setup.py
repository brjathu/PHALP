#!/usr/bin/env python
from setuptools import find_packages, setup
# from pip.req import parse_requirements

setup(
    name='phalp',
    version='0.1.3',    
    description='PHALP: A Python package for People Tracking in 3D',
    url='https://github.com/brjathu/PHALP',
    author='Jathushan Rajasegaran',
    author_email='jathushan@berkeley.edu',
    license='MIT License',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
            "opencv-python",
            "joblib",
            "scikit-learn",
            "pyrender",
            "dill",
            "rich",
            "einops",
            "scenedetect[opencv]",
            "hydra-core",
            "timm",
            "av",
            "smplx==0.1.28",
            "chumpy",           # smplx dependency
            "numpy==1.23.0",    # required by chumpy
            "detectron2 @ git+https://github.com/facebookresearch/detectron2.git",
            "pytube @ git+https://github.com/pytube/pytube.git",
            "pyopengl @ git+https://github.com/mmatl/pyopengl.git",
        ],
    extras_require={
        'all': [
            "hmr2 @ git+https://github.com/shubham-goel/4D-Humans.git@dev",
        ],
        'blur': [
            'facenet_pytorch'
        ]
    },
    dependency_links=[]
)
