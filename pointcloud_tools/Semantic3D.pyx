# distutils: language = c++
# distutils: sources = Semantic3D.cxx

import numpy as np
cimport numpy as np
import cython

cdef extern from "Sem3D.h":
    cdef cppclass Sem3D:
        Sem3D()


        # from PointCloud
        void load_ply(char* filename)
        void load_off(char* filename)
        void save_ply(char* filename)
        void save_off(char* filename)
        void save_ply_mesh(char* filename)
        void save_off_mesh(char* filnename)
        void estimate_normals_hough(int K)
        void estimate_normals_regression(int K)
        void build_mesh()
        int size()
        int size_faces()
        void get_vertices(double* array, int m, int n)
        void get_normals(double* array, int m, int n)
        void get_colors(int* array, int m, int n)
        void get_faces(int* array, int m, int n)
        void set_vertices(double* array, int m, int n)

        # from pointCloud labels
        void load_ply_composite(char* filename)
        void save_ply_composite(char* filename)
        void estimate_noise_radius(float d)
        void estimate_noise_knn(int K)
        void estimate_z_orient()
        void build_mesh(int remove_multi_label_faces)
        void save_mesh_composite(char* filename)
        void get_composite(int* array, int m, int n)
        void get_labels(int* array, int m)
        void set_labels(int* array, int m)

        # from Sem3D
        void set_voxel_size(float vox_size)
        void load_Sem3D(char* filename)
        void load_Sem3D_labels(char* filename, char* labels_filename)
        void load_ply_labels(char* filename)
        void save_ply_labels(char* filename)
        void save_mesh_labels(char* filename)
        void get_labelsColors(int* array, int m, int n)
        void remove_unlabeled_points()
        void mesh_to_label_file_no_labels(char* mesh_filename,	char* sem3d_cloud_txt,	char* output_results)

    # cdef cppclass VoxelsManipulator:
    #     VoxelsManipulator()
    #     void set_vox_size(double size)
    #     int get_size()
    #     void build_from_points(double* array, int n, int feature_size)
    #     void build_from_pointsLabels(double* array, int n, int feature_size)
    #     void get_sub_matrix(double* array, int size, int feature_size, int i_start, int j_start, int k_start, int level);
    #     void points_to_txt(char* filename)
    #     void minimum(double* array);
    #     void maximum(double* array);
    #     void get_random_sub_matrix(char* filename, int size, int level);
    #     void set_matrix_file_scores(char* filename, int level);
    #     void save_txt_scores(char* filename, int save_labels);


cdef class Semantic3D:
    cdef Sem3D sem3d
    def __init__(self):
        self.sem3d = Sem3D()
        # init
        return

    def set_voxel_size(self, vox_size):
        self.sem3d.set_voxel_size(vox_size)

    def load_Sem3D(self, filename):
        return self.sem3d.load_Sem3D(str.encode(filename))

    def load_Sem3D_labels(self, filename, labels_filename):
        return self.sem3d.load_Sem3D_labels(str.encode(filename), str.encode(labels_filename))

    def loadPLYFile(self, filename):
        self.sem3d.load_ply(str.encode(filename))

    def loadPLYFile_composite(self, filename):
        self.sem3d.load_ply_composite(str.encode(filename))


    def loadPLYFile_labels(self, filename):
        self.sem3d.load_ply_labels(str.encode(filename))

    def savePLYFile(self, filename):
        self.sem3d.save_ply(str.encode(filename))

    def savePLYFile_composite(self, filename):
        self.sem3d.save_ply_composite(str.encode(filename))

    def savePLYFile_labels(self, filename):
        self.sem3d.save_ply_labels(str.encode(filename))

    def estimate_normals_hough(self, K):
        self.sem3d.estimate_normals_hough(K)

    def estimate_normals_regression(self, K):
        self.sem3d.estimate_normals_regression(K)

    def estimate_noise_radius(self, d):
        self.sem3d.estimate_noise_radius(d)

    def estimate_noise_knn(self, K):
        self.sem3d.estimate_noise_knn(K)

    def estimate_z_orient(self):
        self.sem3d.estimate_z_orient()

    def build_mesh(self,remove_multi_label_faces):
        self.sem3d.build_mesh(remove_multi_label_faces)

    def save_mesh(self, filename):
        self.sem3d.save_ply_mesh(str.encode(filename))

    def save_mesh_composite(self, filename):
        self.sem3d.save_mesh_composite(str.encode(filename))

    def save_mesh_labels(self, filename):
        self.sem3d.save_mesh_labels(str.encode(filename))

    def size(self):
        return self.sem3d.size()

    def size_faces(self):
        return self.sem3d.size_faces()

    def get_vertices_numpy(self):
        cdef int m, n
        m = self.size()
        n = 3
        d = np.zeros((m,n),dtype=np.double)
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = d
        self.sem3d.get_vertices(<double *> d2.data, m,n)
        return d


    def get_normals_numpy(self):
        cdef int m, n
        m = self.size()
        n = 3
        d = np.zeros((m,n),dtype=np.double)
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = d
        self.sem3d.get_normals(<double *> d2.data, m,n)
        return d

    def get_colors_numpy(self):
        cdef int m, n
        m = self.size()
        n = 3
        d = np.zeros((m,n),dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim = 2] d2 = d
        self.sem3d.get_colors(<int *> d2.data, m,n)
        return d

    def get_composite_numpy(self):
        cdef int m, n
        m = self.size()
        n = 3
        d = np.zeros((m,n),dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim = 2] d2 = d
        self.sem3d.get_composite(<int *> d2.data, m,n)
        return d

    def get_labels_numpy(self):
        cdef int m, n
        m = self.size()
        d = np.zeros(m,dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim = 1] d2 = d
        self.sem3d.get_labels(<int *> d2.data, m)
        return d

    def get_labelsColors_numpy(self):
        cdef int m, n
        m = self.size()
        n = 3
        d = np.zeros((m,n),dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim = 2] d2 = d
        self.sem3d.get_labelsColors(<int *> d2.data, m,n)
        return d

    def get_faces_numpy(self):
        cdef int m, n
        m = self.size_faces()
        n = 3
        d = np.zeros((m,n),dtype=np.int32)
        cdef np.ndarray[np.int32_t, ndim = 2] d2 = d
        self.sem3d.get_faces(<int *> d2.data, m,n)
        return d

    def set_vertices_numpy(self, filename):
        vertices = np.load(filename)["arr_0"].astype(np.double)
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = vertices
        self.sem3d.set_vertices(<double *> d2.data, vertices.shape[0], vertices.shape[1])

    def set_vertices(self, vertices):
        cdef np.ndarray[np.float64_t, ndim = 2] d2 = vertices
        self.sem3d.set_vertices(<double *> d2.data, vertices.shape[0], vertices.shape[1])

    def set_labels_numpy(self, filename):
        labels = np.load(filename)["arr_0"].astype(np.int32)
        cdef np.ndarray[np.int32_t, ndim = 1] d2 = labels
        self.sem3d.set_labels(<int *> d2.data, labels.shape[0])

    cpdef remove_unlabeled_points(self):
        self.sem3d.remove_unlabeled_points()

    cpdef mesh_to_label_file_no_labels(self, mesh_filename,sem3d_cloud_txt,output_results):
        self.sem3d.mesh_to_label_file_no_labels(str.encode(mesh_filename), str.encode(sem3d_cloud_txt), str.encode(output_results))
