# If the pyx file is a C++ file, we should specify that here.
set_source_files_properties(
 ${CMAKE_SOURCE_DIR}/src/Semantic3D.pyx
 PROPERTIES CYTHON_IS_CXX TRUE )

# Multi-file cython modules do not appear to be working at the moment.
cython_add_module( Semantic3D
   Semantic3D.pyx Sem3D.cxx
   pointCloudLabels.cxx
   pointCloud.cxx
   )
target_link_libraries(Semantic3D ${PCL_LIBRARIES})
