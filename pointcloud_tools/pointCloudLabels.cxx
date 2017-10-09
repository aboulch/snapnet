#include "pointCloudLabels.h"

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


PC_Labels::PC_Labels(){
    pc = PointCloudPtr(new PointCloud);
    srand(time(NULL));
}

PC_Labels::~PC_Labels(){
}


void PC_Labels::load_ply_composite(char* filename){
    PointCloudPtr pc_temp(new PointCloud);
    pcl::io::loadPLYFile(filename, *pc_temp);
    noises.resize(pc_temp->size());
    z_orients.resize(pc_temp->size());
    for(size_t pt_id=0; pt_id<pc_temp->size(); pt_id++){
        noises[pt_id] = pc_temp->points[pt_id].r;
        z_orients[pt_id] = pc_temp->points[pt_id].g;
    }
}


void PC_Labels::save_ply_composite(char* filename){
    // create
    vector<Eigen::Vector3i> composite(pc->size());
    #pragma omp parallel for
	for(size_t pt_id=0; pt_id<pc->size(); pt_id++){

        composite[pt_id][0] = int(pc->points[pt_id].r);
        composite[pt_id][1] = int(pc->points[pt_id].g);
        composite[pt_id][2] = int(pc->points[pt_id].b);
        pc->points[pt_id].r = int(noises[pt_id]);
        pc->points[pt_id].g = int(z_orients[pt_id]);
        pc->points[pt_id].b = 0;
	}
    save_ply(filename);
    #pragma omp parallel for
    for(size_t pt_id=0; pt_id<pc->size(); pt_id++){
        pc->points[pt_id].r = composite[pt_id][0];
        pc->points[pt_id].g = composite[pt_id][1];
        pc->points[pt_id].b = composite[pt_id][2];
	}
}

void PC_Labels::estimate_noise_radius(float d)
{
    // Initialize noise vector
    noises = vector<Byte>(pc->size(),0);

    // Create search tree
    pcl::search::KdTree<Point>::Ptr tree (new pcl::search::KdTree<Point>);
    tree->setInputCloud (pc);

    vector<int> point_indices(pc->size());
    #pragma omp parallel for
    for(size_t pt_id=0; pt_id<pc->size(); pt_id++){
        point_indices[pt_id] = pt_id;
    }

    // shuffle indices
    for(size_t pt_id=0; pt_id<pc->size(); pt_id++){
        int temp_pos = rand()%pc->size();
        int temp = point_indices[temp_pos];
        point_indices[temp_pos] = point_indices[pt_id];
        point_indices[pt_id] = temp;
    }

    #pragma omp parallel for
    for(size_t pt_id_t=0; pt_id_t<pc->size(); pt_id_t++){

        int pt_id = point_indices[pt_id_t];

        std::vector<int> pointIdxSearch;
        std::vector<float> pointSquaredDistance;
        tree->radiusSearch(pc->points[pt_id], d, pointIdxSearch, pointSquaredDistance);
        int K = pointIdxSearch.size();

        double a = 0;
        if(K>3){
            Eigen::Matrix3f covariance_matrix;
            Eigen::Vector4f xyz_centroid;
            pcl::computeMeanAndCovarianceMatrix (*pc, pointIdxSearch, covariance_matrix, xyz_centroid);

            // Eigen::Vector4f plane_parameters;
            // float curvature;
            // pcl::solvePlaneParameters (covariance_matrix, xyz_centroid, plane_parameters, curvature);

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(covariance_matrix);

            Eigen::Vector3f ev = eigensolver.eigenvalues();
            //cout << ev.transpose() << endl;
            ev[0] = fabs(ev[0]);
            ev[2] = fabs(ev[2]);
            a = (ev[0]*ev[2] == 0)? 0 : ((ev[0]<ev[2]) ? ev[0] /ev[2] : ev[2] /ev[0]);

            /*//regression

            Eigen::Vector3f mean(0,0,0); // compute mean
            for(int i=0; i<K; i++){
                mean += pc->points[pointIdxSearch[i]].getVector3fMap();
            }
            mean /= K;

            Eigen::Matrix3f cov = Eigen::Matrix3f::Zero(); // covariance matrix
            for(int i=0; i<K; i++){
                Eigen::Vector3f v =  pc->points[pointIdxSearch[i]].getVector3fMap() -mean;
                cov+= (v*v.transpose());
            }

            Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.cast<double>(), Eigen::ComputeFullV);

            double a = svd.singularValues()[2] / svd.singularValues()[0] ;*/

            //a = curvature;
        }
        a /= 0.7;
        a = (a > 1)? 1 : a;
        noises[pt_id] = Byte(a*255);
    }
}

void PC_Labels::estimate_noise_knn(int K)
{
    noises.resize(pc->size());
    // Initialize noise vector
    // noises = vector<Byte>(pc->size(),0);
    //
    //
    // // Create search tree
    // pcl::search::KdTree<Point>::Ptr tree (new pcl::search::KdTree<Point>);
    // tree->setInputCloud (pc);
    //
    // vector<int> point_indices(pc->size());
    // #pragma omp parallel for
    // for(int pt_id=0; pt_id<pc->size(); pt_id++){
    //     point_indices[pt_id] = pt_id;
    // }
    //
    // // shuffle indices
    // for(int pt_id=0; pt_id<pc->size(); pt_id++){
    //     int temp_pos = rand()%pc->size();
    //     int temp = point_indices[temp_pos];
    //     point_indices[temp_pos] = point_indices[pt_id];
    //     point_indices[pt_id] = temp;
    // }
    //
    // std::vector<int> pointIdxSearch(K);
    // std::vector<float> pointSquaredDistance(K);
    // Eigen::Matrix3f covariance_matrix;
    // Eigen::Vector4f xyz_centroid;
    // #pragma omp parallel for shared (tree) private (pointIdxSearch, pointSquaredDistance, covariance_matrix, xyz_centroid)
    // for(int pt_id_t=0; pt_id_t<pc->size(); pt_id_t++){
    //
    //     int pt_id = point_indices[pt_id_t];
    //
    //     // tree->nearestKSearch(pc->points[pt_id], K, pointIdxSearch, pointSquaredDistance);
    //     //
    //     // pcl::computeMeanAndCovarianceMatrix (*pc, pointIdxSearch, covariance_matrix, xyz_centroid);
    //     //
    //     // Eigen::Vector4f plane_parameters;
    //     // float curvature;
    //     // pcl::solvePlaneParameters (covariance_matrix, xyz_centroid, plane_parameters, curvature);
    //
    //     Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(covariance_matrix);
    //     Eigen::Vector3f ev = eigensolver.eigenvalues();
    //     ev[0] = fabs(ev[0]);
    //     ev[2] = fabs(ev[2]);
    //     double a = (ev[0]*ev[2] == 0)? 0 : ((ev[0]<ev[2]) ? ev[0] /ev[2] : ev[2] /ev[0]);
    //
    //
    //     double a = curvature;
    //     a /= 0.7;
    //     a = (a > 1)? 1 : a;
    //     noises[pt_id] = Byte(a*255);
    //
    //
    //
    // }

    float mult = 0.5;
    for(size_t pt_id=0; pt_id<pc->size(); pt_id++){
        float a = fabs(pc->points[pt_id].curvature);
        a = (a > mult)? mult : a;
        noises[pt_id] = Byte(a*255*mult);
    }

}


void PC_Labels::estimate_z_orient()
{
    z_orients.resize(pc->size());
    #pragma omp parallel for
	for(size_t pt_id=0; pt_id<pc->size(); pt_id++){

		Eigen::Vector3f normal = pc->points[pt_id].getNormalVector3fMap();
		normal.normalize();
		double a = acos(fabs(float(normal.dot(Eigen::Vector3f(0,0,1)))));
		a /= M_PI /2.;
        z_orients[pt_id] = Byte(a*255);
	}
}

void PC_Labels::build_mesh(bool remove_multi_label_faces)
{

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

    if(remove_multi_label_faces){
        pcl::PolygonMesh triangles_decimated;
        triangles_decimated.header = triangles.header;
        triangles_decimated.cloud = triangles.cloud;

        for(size_t i=0; i<triangles.polygons.size(); i++){
            const int& v0 = labels[triangles.polygons[i].vertices[0]];
            const int& v1 = labels[triangles.polygons[i].vertices[1]];
            const int& v2 = labels[triangles.polygons[i].vertices[2]];
            if(v0==v1 && v0==v2){
                triangles_decimated.polygons.push_back(triangles.polygons[i]);
            }
        }
        triangles = triangles_decimated;
    }
}


void PC_Labels::save_mesh_composite(char* filename){

    // create composite colors
    vector<Eigen::Vector3i> composite(pc->size());
    #pragma omp parallel for
	for(size_t pt_id=0; pt_id<pc->size(); pt_id++){
        composite[pt_id][0] = int(pc->points[pt_id].r);
        composite[pt_id][1] = int(pc->points[pt_id].g);
        composite[pt_id][2] = int(pc->points[pt_id].b);
        pc->points[pt_id].r = int(noises[pt_id]);
        pc->points[pt_id].g = int(z_orients[pt_id]);
        pc->points[pt_id].b = 0;
	}

    // update mesh colors
    pcl::PCLPointCloud2 pc2;
	pcl::toPCLPointCloud2(*pc, pc2);
	triangles.cloud = pc2;

    // save mesh
    pcl::io::savePolygonFile(filename,triangles);

    // give colors back to point cloud
    #pragma omp parallel for
    for(size_t pt_id=0; pt_id<pc->size(); pt_id++){
        pc->points[pt_id].r = composite[pt_id][0];
        pc->points[pt_id].g = composite[pt_id][1];
        pc->points[pt_id].b = composite[pt_id][2];
	}

}


void PC_Labels::get_composite(int* array, int m, int n)
{
        int i;
    int index = 0 ;

    for (i = 0; i < m; i++) {
        array[index] = noises[i];
        array[index+1] = z_orients[i];
        array[index+2] = 0;
        index += 3;
    }
    return ;
}

void PC_Labels::get_labels(int* array, int m)
{
    int i;
    int index = 0 ;

    for (i = 0; i < m; i++)
    {
        array[index] = labels[i];
        index ++ ;
    }
    return ;
}

void PC_Labels::set_labels(int* array, int m){
    // resize labels
    labels.resize(m);

    int i;
    int index = 0 ;
    for (i = 0; i < m; i++)
    {
        labels[i] = array[index];
        index ++ ;
    }
    return ;
}
