#include "pointCloud.h"

#include "Normals.h"

#include <iostream>
#include <fstream>
#include <map>
#include <Eigen/Dense>
#include <time.h>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/PCLPointCloud2.h>

using namespace std;

PC::PC(): pc(new PointCloud){}

PC::~PC(){}

void PC::load_ply(char* filename){
    pcl::io::loadPLYFile(filename, *pc);
}

void PC::load_off(char* filename){
    // TODO
    cout << "load_off not implemented TODO" << endl;
}

void PC::save_ply(char* filename){
    pcl::io::savePLYFile(filename, *pc);
}

void PC::save_off(char* filename){
    // TODO
    cout << "save_off not implemented TODO" << endl;
}

void PC::save_ply_mesh(char* filename){
    // update mesh colors
    pcl::PCLPointCloud2 pc2;
	pcl::toPCLPointCloud2(*pc, pc2);
	triangles.cloud = pc2;
    pcl::io::savePolygonFile(filename,triangles);
}

void PC::save_off_mesh(char* filename){
    // TODO
    cout << "save_off_mesh not implemented TODO" << endl;
}

void PC::estimate_normals_hough(int K){
    // create the matrix for normal estimation
	Eigen::MatrixX3d points;
	Eigen::MatrixX3d normals;

	points.resize(pc->size(),3);
	for(size_t i=0; i<pc->size(); i++){
		points(i,0) = pc->points[i].x;
		points(i,1) = pc->points[i].y;
		points(i,2) = pc->points[i].z;
	}

	Eigen_Normal_Estimator ne(points, normals);
	ne.neighborhood_size = K;
	ne.estimate_normals();

	for(size_t i=0; i<pc->size(); i++){
		pc->points[i].normal_x = normals(i,0);
		pc->points[i].normal_y = normals(i,1);
		pc->points[i].normal_z = normals(i,2);
	}
}

void PC::estimate_normals_regression(int K){
    // Create the normal estimation class, and pass the input dataset to it
    pcl::NormalEstimationOMP<Point, Point> ne;
    ne.setInputCloud (pc);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<Point>::Ptr tree (new pcl::search::KdTree<Point> ());
    ne.setSearchMethod (tree);

    // Use all neighbors in a sphere of radius 3cm
    //ne.setRadiusSearch (0.03);
    ne.setKSearch (K);

    // Compute the features
    ne.compute (*pc);
}

void PC::build_mesh(){

    // Create search tree
    pcl::search::KdTree<Point>::Ptr tree (new pcl::search::KdTree<Point>);
    tree->setInputCloud (pc);

    // Initialize objects
    pcl::GreedyProjectionTriangulation<Point> gp3;

    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius (2);

    // Set typical values for the parameters
    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors (100);
    gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
    gp3.setMinimumAngle(M_PI/18); // 10 degrees
    gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
    gp3.setNormalConsistency(true);

    // Get result
    gp3.setInputCloud (pc);
    gp3.setSearchMethod (tree);
    gp3.reconstruct (triangles);
}

int PC::size(){
    return pc->size();
}

int PC::size_faces(){
    return triangles.polygons.size();
}

void PC::get_vertices(double* array, int m, int n) {

    int i, j ;
    int index = 0 ;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            array[index] = pc->points[i].getVector3fMap()[j];
            index ++ ;
            }
        }
    return ;
}

void PC::get_normals(double* array, int m, int n) {

    int i, j ;
    int index = 0 ;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            array[index] = pc->points[i].getNormalVector3fMap()[j];
            index ++ ;
            }
        }
    return ;
}

void PC::get_colors(int* array, int m, int n){
    int i, j ;
    int index = 0 ;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            array[index] = pc->points[i].getRGBVector3i()[j];
            index ++ ;
            }
        }
    return ;
}

void PC::get_faces(int* array, int m, int n){
    int i, j ;
    int index = 0 ;
    for (i = 0; i < m; i++) {
        for(j=0; j<n; j++){
            array[index] = triangles.polygons[i].vertices[j];
            index++;
        }
    }
    return ;
}

void PC::set_vertices(double* array, int m, int n){
    // resize the point cloud
    pc->resize(m);

    // fill the point cloud
    int i, j ;
    int index = 0 ;
    for (i = 0; i < m; i++) {
        Eigen::Vector3f pt;
        for (j = 0; j < n; j++) {
            pt[j] = array[index];
            index ++ ;
        }
        pc->points[i].getVector3fMap() = pt;
    }
    return ;
}
