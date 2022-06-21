import pathlib
from setuptools import setup, find_packages
import os

HERE = pathlib.Path(__file__).parent

VERSION = '2.0.6'
PACKAGE_NAME = 'hawk_eyes'
AUTHOR = 'Duy Nguyen Manh'
AUTHOR_EMAIL = 'manhduy160396@email.com'
URL = 'https://github.com/Duy-NM/hawk_eye'

LICENSE = 'MIT License'
DESCRIPTION = 'Face recognize, Object tracking, OCR'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'opencv-python',
      'gdown',
      'onnx',
      'onnxruntime-gpu==1.7'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )
