# distutils: language = c++
# distutils: sources = Semantic3D.cxx

cimport c_sem3D
import numpy as np
cimport numpy as np
import cython

cdef class Sem3D:
    cdef c_sem3D.Sem3D *thisptr

    def __cinit__(self):
        self.thisptr = new c_sem3D.Sem3D()

    def __dealloc__(self):
        del self.thisptr

    cpdef set_voxel_size(self, vox_size):
        self.thisptr.set_voxel_size(vox_size)

    cpdef load_Sem3D(self, filename):
        return self.thisptr.load_Sem3D(str.encode(filename))

    cpdef load_Sem3D_labels(self, filename, labels_filename):
        return self.thisptr.load_Sem3D_labels(str.encode(filename), str.encode(labels_filename))

    cpdef loadPLYFile(self, filename):
        self.thisptr.load_ply(str.encode(filename))

    cpdef loadPLYFile_composite(self, filename):
        self.thisptr.load_ply_composite(str.encode(filename))


    cpdef loadPLYFile_labels(self, filename):
        self.thisptr.load_ply_labels(str.encode(filename))

    cpdef savePLYFile(self, filename):
        self.thisptr.save_ply(str.encode(filename))

    cpdef savePLYFile_composite(self, filename):
        self.thisptr.save_ply_composite(str.encode(filename))

    cpdef savePLYFile_labels(self, filename):
        self.thisptr.save_ply_labels(str.encode(filename))

    cpdef estimate_normals_hough(self, K):
        self.thisptr.estimate_normals_hough(K)

    cpdef estimate_normals_regression(self, K):
        self.thisptr.estimate_normals_regression(K)

    cpdef estimate_noise_radius(self, d):
        self.thisptr.estimate_noise_radius(d)

    cpdef estimate_noise_knn(self, K):
        self.thisptr.estimate_noise_knn(K)

    cpdef estimate_z_orient(self):
        self.thisptr.estimate_z_orient()

    cpdef build_mesh(self,remove_multi_label_faces):
        self.thisptr.build_mesh(remove_multi_label_faces)

    cpdef save_mesh(self, filename):
        self.thisptr.save_ply_mesh(str.encode(filename))

    cpdef save_mesh_composite(self, filename):
        self.thisptr.save_mesh_composite(str.encode(filename))

    cpdef save_mesh_labels(self, filename):
        self.thisptr.save_mesh_labels(str.encode(filename))

    cpdef size(self):
        return self.thisptr.size()

    cpdef size_faces(self):
        return self.thisptr.size_faces()

    def get_vertices_numpy(self):
        cdef int m, n
        m = self.size()
        n = 3
        d = np.zeros((m,n),dtype=np.double)
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = d
        self.thisptr.get_vertices(<double *> d2.data, m,n)
        return d


    def get_normals_numpy(self):
        cdef int m, n
        m = self.size()
        n = 3
        d = np.zeros((m,n),dtype=np.double)
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = d
        self.thisptr.get_normals(<double *> d2.data, m,n)
        return d

    def get_colors_numpy(self):
        cdef int m, n
        m = self.size()
        n = 3
        d = np.zeros((m,n),dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim = 2] d2 = d
        self.thisptr.get_colors(<int *> d2.data, m,n)
        return d

    def get_composite_numpy(self):
        cdef int m, n
        m = self.size()
        n = 3
        d = np.zeros((m,n),dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim = 2] d2 = d
        self.thisptr.get_composite(<int *> d2.data, m,n)
        return d

    def get_labels_numpy(self):
        cdef int m, n
        m = self.size()
        d = np.zeros(m,dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim = 1] d2 = d
        self.thisptr.get_labels(<int *> d2.data, m)
        return d

    def get_labelsColors_numpy(self):
        cdef int m, n
        m = self.size()
        n = 3
        d = np.zeros((m,n),dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim = 2] d2 = d
        self.thisptr.get_labelsColors(<int *> d2.data, m,n)
        return d

    def get_faces_numpy(self):
        cdef int m, n
        m = self.size_faces()
        n = 3
        d = np.zeros((m,n),dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim = 2] d2 = d
        self.thisptr.get_faces(<int *> d2.data, m,n)
        return d

    def set_vertices_numpy(self, filename):
        vertices = np.load(filename)["arr_0"].astype(np.double)
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = vertices
        self.thisptr.set_vertices(<double *> d2.data, vertices.shape[0], vertices.shape[1])

    def set_vertices(self, vertices):
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = vertices
        self.thisptr.set_vertices(<double *> d2.data, vertices.shape[0], vertices.shape[1])

    def set_labels_numpy(self, filename):
        labels = np.load(filename)["arr_0"].astype(np.int32)
        cdef np.ndarray[np.int32_t, ndim = 1] d2 = labels
        self.thisptr.set_labels(<int *> d2.data, labels.shape[0])

    cpdef remove_unlabeled_points(self):
        self.thisptr.remove_unlabeled_points()

    cpdef mesh_to_label_file_no_labels(self, mesh_filename,sem3d_cloud_txt,output_results):
        self.thisptr.mesh_to_label_file_no_labels(str.encode(mesh_filename), str.encode(sem3d_cloud_txt), str.encode(output_results))
