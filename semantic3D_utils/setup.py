from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

pcl_include_dir = "/usr/local/include/pcl-1.8"
pcl_lib_dir = "/usr/local/lib"
vtk_include_dir = "/usr/include/vtk-6.2"

ext_modules = [Extension(
       "semantic3D",
       sources=["Semantic3D.pyx", "Sem3D.cxx",],  # source file(s)
       include_dirs=["./", numpy.get_include(),pcl_include_dir, vtk_include_dir],
       language="c++",
       library_dirs=[pcl_lib_dir],
       libraries=["pcl_common","pcl_kdtree","pcl_features","pcl_surface","pcl_io"],         
       extra_compile_args = [ "-std=c++11", "-fopenmp",],
       extra_link_args=["-std=c++11", '-fopenmp'],
  )]

setup(
    name = "Semantic3D_utils",
    ext_modules = ext_modules,
    cmdclass = {'build_ext': build_ext},
)
