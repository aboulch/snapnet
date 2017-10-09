#ifndef _Sem3D_h
#define _Sem3D_h

#include <string>
#include <vector>

#include <Eigen/Dense>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/surface/gp3.h>

#include "pointCloudLabels.h"

class Sem3D: public PC_Labels
{
public:

    //#################################
    //#################################
    float voxel_size;

    Sem3D(); // empty constructor

    void set_voxel_size(float vox_size);

    // IO methods
    void load_Sem3D(char* filename);
    void load_Sem3D_labels(char* filename, char* labels_filename);
    void load_ply_labels(char* filename);
    void save_ply_labels(char* filename);

    // mesh creation
    void save_mesh_labels(char* filename);

    void get_labelsColors(int* array, int m, int n);

    void remove_unlabeled_points();

    void mesh_to_label_file_no_labels(char* mesh_filename,
    		char* sem3d_cloud_txt,
    		char* output_results);

    //#################################
    //#################################
    ~Sem3D();

};

#endif
