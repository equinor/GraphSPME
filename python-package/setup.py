import os
from glob import glob
from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "graph_spme",
        sorted(glob("src/*.cpp")),
        cxx_std=14,
        include_dirs=[os.path.join(os.path.dirname(__file__),"../include/graph_spme")],
        # Example: passing in the version to the compiled code
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

setup(
    name="GraphSPME",
    version=__version__,
    author="Berent Ånund Strømnes Lunde",
    author_email="lundeberent@gmail.com",
    url="https://github.com/Blunde1/GraphSPME",
    description="High dimensional precision matrix estimation with a known graphical structure",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    zip_safe=False,
    python_requires=">=3.6",
)
