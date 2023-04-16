from setuptools import setup, find_packages
import os
import pkg_resources
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text()
DESCRIPTION = 'ImageReward'

# 配置
setup(
        name="image-reward", 
        py_modules = ["ImageReward"],
        version="1.0",
        author="Jiazheng Xu, et al.",
        author_email="<xjz22@mails.tsinghua.edu.cn>",
        url="https://github.com/THUDM/ImageReward",
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type='text/markdown',
        packages=find_packages(exclude=["tests*"]),
        install_requires=[
            'timm==0.6.13',
            'transformers==4.27.4',
            'fairscale==0.4.13',
            'huggingface_hub==0.13.4',
        ],
        dependency_links = [
            "clip @ git+https://github.com/openai/CLIP.git",
        ],
        python_requires=">=3.5",
        license="Apache 2.0 license"
)
