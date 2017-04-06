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
    bool load_Sem3D(const std::string& filename);
    bool load_Sem3D_labels(const std::string& filename, const std::string& labels_filename);
    void load_ply_labels(const std::string& filename);
    void save_ply_labels(const std::string& filename);

    // mesh creation
    void save_mesh_labels(const std::string& filename);

    void get_labelsColors(int* array, int m, int n);

    void remove_unlabeled_points();

    void mesh_to_label_file_no_labels(const std::string& mesh_filename,
    		const std::string& sem3d_cloud_txt,
    		const std::string& output_results);

    //#################################
    //#################################
    ~Sem3D();

};

#endif
