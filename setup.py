#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name='phalp',
    version='0.1.0',    
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



# """Setup PHALP."""

# from setuptools import find_packages, setup


# import ipdb; ipdb.set_trace()
# setup(
#     name="phalp",
#     version="0.1.0",
#     description='PHALP: A Python package for People Tracking in 3D',
#     long_description_content_type="text/markdown",
#     url="https://github.com/facebookresearch/pycls",
#     packages=find_packages(),
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "License :: OSI Approved :: MIT License",
#     ],
#     install_requires=["numpy", "opencv-python", "simplejson", "yacs"],
# )