# distutils: language = c++
# distutils: sources = Semantic3D.cxx

import numpy as np
cimport numpy as np
import cython

cdef extern from "Sem3D.h":

    void project_labels_to_point_cloud(
	char* output_filename,
	char* filename_pc, 
	char* filename_pc_with_labels, 
	char* filename_labels)

    void sem3d_from_txt_voxelize(char* filename, char* destination_filename, float voxel_size)
    void sem3d_from_txt_voxelize_labels(char* filename, char* labels_filename, char* destination_filename, float voxel_size)
    void sem3d_estimate_attributes(char* filename, char* destination_filename, float normals_k)
    void sem3d_create_mesh(char* filename_rgb,
						char* filename_composite,
						char* filename_mesh_rgb, 
						char* filename_mesh_composite,
						char* filename_mesh_label, 
                        char* filename_faces, 
						int remove_multi_label_faces)

def project_labels_to_pc(
    output_filename,
    input_points_filename,
    reference_points_filename,
    reference_labels_filename):

    project_labels_to_point_cloud(output_filename.encode(),
                         input_points_filename.encode(),
                         reference_points_filename.encode(),
                         reference_labels_filename.encode())

def semantic3d_load_from_txt_voxel(filename, filename_dest, voxel_size):
    cdef bytes filename_bytes = filename.encode()
    cdef bytes filename_dest_bytes = filename_dest.encode()
    sem3d_from_txt_voxelize(filename_bytes, filename_dest_bytes, voxel_size)

def semantic3d_load_from_txt_voxel_labels(filename, filename_labels, filename_dest, voxel_size):
    cdef bytes filename_bytes = filename.encode()
    cdef bytes filename_labels_bytes = filename_labels.encode()
    cdef bytes filename_dest_bytes = filename_dest.encode()
    sem3d_from_txt_voxelize_labels(filename_bytes, filename_labels_bytes, filename_dest_bytes, voxel_size)

def semantic3d_estimate_attributes(filename, filename_dest, K):
    cdef bytes filename_bytes = filename.encode()
    cdef bytes filename_dest_bytes = filename_dest.encode()
    sem3d_estimate_attributes(filename_bytes, filename_dest_bytes, K)

def semantic3d_create_mesh(
                        filename_rgb,
						filename_composite,
						filename_mesh_rgb, 
						filename_mesh_composite,
						filename_mesh_label, 
                        filename_faces,
						remove_multi_label_faces=False):
    cdef bytes filename_rgb_bytes = filename_rgb.encode()
    cdef bytes filename_composite_bytes = filename_composite.encode()
    cdef bytes filename_mesh_rgb_bytes = filename_mesh_rgb.encode()
    cdef bytes filename_mesh_composite_bytes = filename_mesh_composite.encode()
    cdef bytes filename_mesh_label_bytes = filename_mesh_label.encode()
    cdef bytes filename_faces_bytes = filename_faces.encode()
    
    sem3d_create_mesh(filename_rgb_bytes,
                        filename_composite_bytes,
                        filename_mesh_rgb_bytes,
                        filename_mesh_composite_bytes,
                        filename_mesh_label_bytes, 
                        filename_faces_bytes,
                        remove_multi_label_faces)
