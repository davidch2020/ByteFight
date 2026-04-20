from setuptools import Extension, setup


# Build the optional native search module for this refactored CPP experiment.
# The extension name stays the same so agent.py can load it without extra config.
experiment_extensions = [
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
    ext_modules=experiment_extensions,
)
