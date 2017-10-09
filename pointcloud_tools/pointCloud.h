#ifndef _pointCloud_h
#define _pointCloud_h

#include <string>
#include <vector>

#include <Eigen/Dense>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/surface/gp3.h>



class PC
{

protected:
    // type definition
    typedef pcl::PointXYZRGBNormal Point;
    typedef pcl::Normal Normal;
    typedef pcl::PointCloud<Point> PointCloud;
    typedef pcl::PointCloud<Normal> NormalCloud;
    typedef PointCloud::Ptr PointCloudPtr;
    typedef unsigned char Byte;

    //#################################
    //#################################
    PointCloudPtr pc;
    pcl::PolygonMesh triangles; // a triangular mesh

public:
    //#################################
    //#################################

    PC(); // empty constructor
    ~PC(); // destructor

    //#################################
    //#################################
    // IO methods
    void load_ply(char* filename);
    void load_off(char* filename);

    void save_ply(char* filename);
    void save_off(char* filename);
    void save_ply_mesh(char* filename);
    void save_off_mesh(char* filnename);

    // normal estimation
    void estimate_normals_hough(int K);
    void estimate_normals_regression(int K);

    // mesh creation
    void build_mesh(); // to be overloaded if labeled pointcloud

    // access methods
    int size();
    int size_faces();

    // for numpy copy access
    void get_vertices(double* array, int m, int n);
    void get_normals(double* array, int m, int n);
    void get_colors(int* array, int m, int n);
    void get_faces(int* array, int m, int n);

    // for numpy
    void set_vertices(double* array, int m, int n);
};

#endif
