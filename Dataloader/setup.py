from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "prefetch",
        ["prefetch.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=["-O3", "-march=native"]
    )
]

setup(
    name="prefetch",
    ext_modules=ext_modules,
    zip_safe=False,
)
