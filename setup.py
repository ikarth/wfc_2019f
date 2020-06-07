import os
from setuptools import setup

PROJECT_DIR = os.path.dirname(__file__)

def read(fname):
    return open(os.path.join(PROJECT_DIR, fname)).read()


version = '0.1'


setup(
    name="wfc_python",
    version=version,
    author = "Isaac Karth",
    author_email = "isaac@isaackarth.com",
    description = "Implementation of wave function collapse in Python",
    license = "MIT License",
    keywords = "sample wfc wave-function-collapse",
    url = "https://github.com/ikarth/wfc_python",
    packages=['wfc', 'tests'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
    ],
)