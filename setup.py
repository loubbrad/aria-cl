import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="ariacl",
    py_modules=["ariacl"],
    version="0.0.1",
    description="",
    author="",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "aria-pianocl=ariacl.run:main",
        ],
    },
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
)