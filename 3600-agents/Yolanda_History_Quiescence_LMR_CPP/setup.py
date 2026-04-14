from setuptools import Extension, setup


# Minimal native build for Yolanda's optional search backend.
# We keep this tiny on purpose so the first .so is easy to compile and test.
ext_modules = [
    Extension(
        "yolanda_search_ext",
        ["yolanda_search_ext.cpp"],
        language="c++",
        extra_compile_args=["-O3"],
    )
]


setup(
    name="yolanda_search_ext",
    version="0.1.0",
    ext_modules=ext_modules,
)
