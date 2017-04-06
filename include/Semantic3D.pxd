cimport c_sem3D
from c_sem3D cimport Sem3D as CSem3D

cdef class Sem3D:
    cdef c_sem3D.Sem3D *thisptr      # hold a C++ instance which we're wrapping

    cpdef set_voxel_size(self, vox_size)
    cpdef load_Sem3D(self, filename)
    cpdef load_Sem3D_labels(self, filename, labels_filename)

    cpdef loadPLYFile(self, filename)
    cpdef loadPLYFile_composite(self, filename)
    cpdef loadPLYFile_labels(self, filename)

    cpdef savePLYFile(self, filename)
    cpdef savePLYFile_composite(self, filename)
    cpdef savePLYFile_labels(self, filename)

    cpdef estimate_normals_hough(self, K)
    cpdef estimate_normals_regression(self, K)
    cpdef estimate_normals_regression_depth(self, K)
    cpdef estimate_noise_radius(self,d)
    cpdef estimate_noise_knn(self, K)
    cpdef estimate_z_orient(self)
    cpdef build_mesh(self,remove_multi_label_faces)
    cpdef save_mesh(self, filename)
    cpdef save_mesh_composite(self, filename)
    cpdef save_mesh_labels(self, filename)

    cpdef size(self)
    cpdef size_faces(self)

    cpdef remove_unlabeled_points(self)

    cpdef mesh_to_label_file_no_labels(self, mesh_filename, sem3d_cloud_txt, output_results)
