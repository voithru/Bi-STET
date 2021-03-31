from setuptools import setup, find_packages, find_namespace_packages

_dependency = [
        "pillow>=5.4.1",
        "nltk==3.4.5",
        "numpy==1.17.1",
        "scipy==1.2.0",
        'seaborn==0.9.0',
        "tensorboard-logger==0.1.0",
        "tensorboardX>=1.7",
        "torch>=1.1.0",
        "torchvision>=0.2.1",
        "transformers==2.1.1"
    ]


setup(name='Kobistet',
      version='0.1',
      description='bistet for korean ocr',
      author='work82mj',
      author_email='work82mj''@''gmail.com',
      license="unlicense",
      # packages=find_namespace_packages(),
      packages=find_packages(),
      python_requires=">=3.7.0",
      install_requires=[],
      extras_require={'all':_dependency},
      include_package_data=True,
      zip_safe=False)