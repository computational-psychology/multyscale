from setuptools import setup, find_packages

setup(
    name="multyscale",
    version="0.2",
    description="Multiscale spatial filtering",
    author="Joris Vincent",
    author_email="joris.vincent@tu-berlin.de",
    license="MIT",
    package_dir={"stimuli": "stimuli"},
    packages=find_packages(),
    zip_safe=False,
)
