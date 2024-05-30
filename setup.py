from setuptools import setup, find_packages  # type: ignore
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
LONG_DESCRIPTION = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="j2a",
    version="0.1",
    packages=find_packages(),
    url="",
    license="",
    author="Anders Lauridsen, Jacob Mørk, & Jakob Olsen",
    author_email="",
    description="",
)
