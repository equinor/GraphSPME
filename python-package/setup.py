import os
from glob import glob
from setuptools import find_packages, setup
from setuptools_scm import get_version

from pybind11.setup_helpers import Pybind11Extension

__version__ = get_version(root=os.path.join(os.path.dirname(__file__), ".."))

ext_modules = [
    Pybind11Extension(
        "_graphspme",
        sorted(glob("src/*.cpp")),
        cxx_std=14,
        include_dirs=[
            os.path.join(os.path.dirname(__file__), "../include/graph_spme"),
            "/usr/include/eigen3",
            "/usr/local/include/eigen3",
            "../eigen",
        ],
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

with open("README.md") as f:
    long_description = f.read()

setup(
    name="GraphSPME",
    version=__version__,
    author="Berent Ånund Strømnes Lunde",
    author_email="lundeberent@gmail.com",
    url="https://github.com/Blunde1/GraphSPME",
    description="High dimensional precision matrix estimation with a known graphical structure",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    install_requires=["numpy", "scipy"],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    extras_require={"test": ["pytest"]},
    zip_safe=False,
    python_requires=">=3.6",
)
