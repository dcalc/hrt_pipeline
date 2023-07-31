from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import platform

compile_extra_args = []
link_extra_args = []

# To be modified by user if necessary
# import platform
if platform.system() == "Windows":
    compile_extra_args = ["/std:c++latest", "/EHsc"]
elif platform.system() == "Darwin":
    #pass
    # compile_extra_args = ["-mmacosx-version-min=10.13"]
    # link_extra_args = ["-mmacosx-version-min=10.13"]
    compile_extra_args = ["-mmacosx-version-min=13.00"]
    link_extra_args = ["-mmacosx-version-min=13.00"]


ext_modules = Extension(
    name="pymilos",
    sources=["pymilos.pyx"],
    libraries=["milos"],
    library_dirs=["lib"],
    include_dirs=["lib",numpy.get_include()],
    extra_compile_args = compile_extra_args,
    extra_link_args = link_extra_args,
    # define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    )
setup(
    name="pymilos",
    ext_modules=cythonize([ext_modules])
)
