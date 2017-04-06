from libcpp.string cimport string
import numpy as np
cimport numpy as np

cdef extern from "Sem3D.h":
    cdef cppclass Sem3D:

        Sem3D()
        void set_voxel_size(float)
        int load_Sem3D(string)
        int load_Sem3D_labels(string,string)

        void load_ply(string)
        void load_ply_composite(string)
        void load_ply_labels(string)
        void save_ply_labels(string)
        void save_ply(string)
        void save_ply_composite(string)

        void estimate_normals_hough(int)
        void estimate_normals_regression(int)
        void estimate_noise_radius(float)
        void estimate_noise_knn(int)
        void estimate_z_orient()

        void build_mesh(int)
        void save_ply_mesh(string)
        void save_mesh_composite(string)
        void save_mesh_labels(string)

        void get_vertices(double*, int,int)
        void get_normals(double*, int,int)
        void get_colors(int*, int, int)
        void get_composite(int*, int, int)
        void get_labels(int*, int)
        void get_labelsColors(int*, int, int)
        void get_faces(int*, int,int)

        void set_vertices(double*, int, int)
        void set_labels(int*, int)

        int size()
        int size_faces()

        void remove_unlabeled_points()

        void mesh_to_label_file_no_labels(string, string, string)
