#ifndef POINTCLOUDLABELS_HEADER
#define POINTCLOUDLABELS_HEADER

#include <string>
#include <vector>

#include <Eigen/Dense>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/surface/gp3.h>

#include "pointCloud.h"

class PC_Labels: public PC
{
public:

    //#################################
    //#################################
    std::vector<Byte> labels;
    std::vector<Byte> noises;
    std::vector<Byte> z_orients;

    PC_Labels(); // empty constructor

    // IO methods
    void load_ply_composite(char* filename);
    void save_ply_composite(char* filename);

    // noise estimation
    void estimate_noise_radius(float d);
    void estimate_noise_knn(int K);
    void estimate_z_orient();

    // mesh creation
    void build_mesh(bool remove_multi_label_faces=false);
    void save_mesh_composite(char* filename);

    void get_composite(int* array, int m, int n);
    void get_labels(int* array, int m);
    void set_labels(int* array, int m);

    //#################################
    //#################################
    ~PC_Labels();

};

#endif
