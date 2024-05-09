from setuptools import setup, find_packages
import os.path

setup(
    name="comfy_bria_rmbg",
    version="0.0.1",
    packages=find_packages(),
    install_requires=open(os.path.join(os.path.dirname(__file__), "requirements.txt")).readlines(),
    author='',
    author_email='',
    description='',
    entry_points={
        'comfyui.custom_nodes': [
            'comfy_bria_rmbg = comfy_bria_rmbg',
        ],
    },
)