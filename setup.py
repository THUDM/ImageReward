from setuptools import setup, find_packages
import os
import pkg_resources
from pathlib import Path

long_description = (Path(__file__).parent / "README-pypi.md").read_text()
DESCRIPTION = 'ImageReward'

# 配置
setup(
        name="image-reward", 
        py_modules = ["ImageReward"],
        version="1.0",
        author="xujz18",
        author_email="<xjz22@mails.tsinghua.edu.cn>",
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type='text/markdown',
        packages=find_packages(exclude=["tests*"]),
        install_requires=[
            str(r)
            for r in pkg_resources.parse_requirements(
                open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
            )
        ],
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
)
