# Copyright (c) 2019 Eric Steinberger


import os.path as osp

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DeepCFR",
    version="0.0.1",
    author="Eric Steinberger",
    author_email="ericsteinberger.est@gmail.com",
    description="A scalable implementation of Deep CFR and its successor Single Deep CFR (SD-CFR) in the PokerRL framework.",
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ErocSteinberger/Deep-CFR",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
)
