#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"

typedef std::vector<std::vector<double> > my_vector_of_vectors_t;



#include "Normals.h"
#include "Sem3D.h"
#include "Eigen/Dense"
#include <map>
#include <time.h>

using namespace std;

// class for voxels
class Voxel_center{
public:
    float x,y,z,d;
    int r,g,b;
    int intensity;
    int label;
};

// comparator for voxels
struct Vector3icomp {
	bool operator() (const Eigen::Vector3i& v1, const Eigen::Vector3i& v2) const{
		if(v1[0] < v2[0]){
			return true;
		}else if(v1[0] == v2[0]){
			if(v1[1] < v2[1]){
				return true;
			}else if(v1[1] == v2[1] && v1[2] < v2[2]){
				return true;
			}
		}
		return false;
	}
};

void sem3d_from_txt_voxelize(char* filename, char* destination_filename, float voxel_size){

    // open the semantic 3D file
    std::ifstream ifs(filename);
    std::string line;
    int pt_id =0;

    std::map<Eigen::Vector3i, Voxel_center, Vector3icomp> voxels;
    while(getline(ifs,line)){
        pt_id++;

        if(pt_id%100000==0){
            cout << "\r";
            cout << pt_id /1000000. << "M points loaded"<< std::flush;
        }

        std::stringstream sstr(line);
        float x,y,z;
        float intensity;
        int r, g, b;
        sstr >> x >> y >> z >> intensity >> r >> g >> b;

        int x_id = std::floor(x/voxel_size) + 0.5; // + 0.5, centre du voxel (k1*res, k2*res)
        int y_id = std::floor(y/voxel_size) + 0.5;
        int z_id = std::floor(z/voxel_size) + 0.5;

        Eigen::Vector3i vox(x_id, y_id, z_id);
        double d = (x-x_id)*(x-x_id) + (y-y_id)*(y-y_id) + (z-z_id)*(z-z_id);

        if(voxels.count(vox)>0){
            const Voxel_center& vc_ = voxels[vox];
            if(vc_.d > d){
                Voxel_center vc;
                vc.x = std::floor(x/voxel_size)*voxel_size;
                vc.y = std::floor(y/voxel_size)*voxel_size;
                vc.z = std::floor(z/voxel_size)*voxel_size;
                vc.d = d;
                vc.r = r;
                vc.g = g;
                vc.b = b;
                vc.intensity = intensity;
                voxels[vox] = vc;
            }

        }else{
            Voxel_center vc;
            vc.x = std::floor(x/voxel_size)*voxel_size;
            vc.y = std::floor(y/voxel_size)*voxel_size;
            vc.z = std::floor(z/voxel_size)*voxel_size;
            vc.d = d;
            vc.r = r;
            vc.g = g;
            vc.b = b;
            vc.intensity = intensity;
            voxels[vox] = vc;
        }
    }
    ifs.close();
    cout << endl;

    ofstream ofs (destination_filename);
    for(std::map<Eigen::Vector3i, Voxel_center>::iterator it=voxels.begin(); it != voxels.end(); it++){
        ofs << it->second.x << " ";
        ofs << it->second.y << " ";
        ofs << it->second.z << " ";
        ofs << it->second.r << " ";
        ofs << it->second.g << " ";
        ofs << it->second.b << " ";
        ofs << "0" << endl; // for the label
    }
    ofs.close();
}

void sem3d_from_txt_voxelize_labels(char* filename, char* labels_filename,
    char* destination_filename, float voxel_size){

    std::ifstream ifs(filename);
	std::ifstream ifs_labels(labels_filename);
	std::string line;
	std::string line_labels;
	int pt_id =0;

	std::map<Eigen::Vector3i, Voxel_center, Vector3icomp> voxels;
	while(getline(ifs,line)){
		pt_id++;
		getline(ifs_labels, line_labels);


        if(pt_id%100000==0){
            cout << "\r";
            cout << pt_id /1000000.  << "M points loaded"<< std::flush;
        }

		std::stringstream sstr_label(line_labels);
		int label;
		sstr_label >> label;

		// continue if points is unlabeled
		if(label == 0)
			continue;


		std::stringstream sstr(line);
		float x,y,z;
		int intensity;
		int r, g, b;
		sstr >> x >> y >> z >> intensity >> r >> g >> b;

		int x_id = std::floor(x/voxel_size) + 0.5; // + 0.5, centre du voxel (k1*res, k2*res)
		int y_id = std::floor(y/voxel_size) + 0.5;
		int z_id = std::floor(z/voxel_size) + 0.5;

		Eigen::Vector3i vox(x_id, y_id, z_id);
		double d = (x-x_id)*(x-x_id) + (y-y_id)*(y-y_id) + (z-z_id)*(z-z_id);

		if(voxels.count(vox)>0){
			const Voxel_center& vc_ = voxels[vox];
			if(vc_.d > d){
				Voxel_center vc;
				vc.x = std::floor(x/voxel_size)*voxel_size;
				vc.y = std::floor(y/voxel_size)*voxel_size;
				vc.z = std::floor(z/voxel_size)*voxel_size;
				vc.d = d;
				vc.r = r;
				vc.g = g;
				vc.b = b;
				vc.intensity = intensity;
				vc.label = label;
				voxels[vox] = vc;
			}

		}else{
			Voxel_center vc;
			vc.x = std::floor(x/voxel_size)*voxel_size;
			vc.y = std::floor(y/voxel_size)*voxel_size;
			vc.z = std::floor(z/voxel_size)*voxel_size;
			vc.d = d;
			vc.r = r;
			vc.g = g;
			vc.b = b;
			vc.intensity = intensity;
			vc.label = label;
			voxels[vox] = vc;
		}
	}
    ifs.close();
    ifs_labels.close();
    cout << endl;
    ofstream ofs (destination_filename);
    for(std::map<Eigen::Vector3i, Voxel_center>::iterator it=voxels.begin(); it != voxels.end(); it++){
        ofs << it->second.x << " ";
        ofs << it->second.y << " ";
        ofs << it->second.z << " ";
        ofs << it->second.r << " ";
        ofs << it->second.g << " ";
        ofs << it->second.b << " ";
        ofs << it->second.label << endl; // for the label
    }
    ofs.close();
}

void project_labels_to_point_cloud(
	char* output_filename,
	char* filename_pc, 
	char* filename_pc_with_labels, 
	char* filename_labels)
{
	// open the file stream
	ifstream ifs_labels(filename_labels);
	ifstream ifs_pts(filename_pc_with_labels);

	// load the points and labels
	my_vector_of_vectors_t  samples;
	std::vector<int> labels;
	string line;
	string line_label;
	cout << "Getting labeled points..." << endl;
	while(std::getline(ifs_pts, line)){

		// get the coordinates
		std::istringstream iss(line);
        float x,y,z;
        iss >> x >> y >> z;

		// get the label
		std::getline(ifs_labels, line_label);
		std::istringstream iss_label(line_label);
		int label;
		iss_label >> label;

		vector<double> point(3);
		point[0] = x;
		point[1] = y;
		point[2] = z;

		samples.push_back(point);
		labels.push_back(label);
	}
	ifs_labels.close();
	ifs_pts.close();
	cout << "Done " << samples.size() << " points" << endl;

	cout << "Create KDTree..." << endl;
	size_t dim=3;
	typedef KDTreeVectorOfVectorsAdaptor< my_vector_of_vectors_t, double >  my_kd_tree_t;
	my_kd_tree_t   mat_index(dim /*dim*/, samples, 10 /* max leaf */ );
	mat_index.index->buildIndex();
	cout << "Done" << endl;

	cout << "Iteration on the original point cloud..." << endl; 
	ifstream ifs(filename_pc);
	ofstream ofs(output_filename);
    int pt_id = 0;
    while (std::getline(ifs, line))
    {   
		if((pt_id+1)%1000000==0){
			cout << "\r                              \r";
			cout << (pt_id+1)/1000000 << " M";
		}

		// get the query point coordinates
		std::istringstream iss(line);
        float x,y,z;
        iss >> x >> y >> z;
		std::vector<double> query_pt(3);
		query_pt[0] = x;
		query_pt[1] = y;
		query_pt[2] = z;

		// search for nearest neighbor
		const size_t num_results = 1;
		std::vector<size_t>   ret_indexes(num_results);
		std::vector<double> out_dists_sqr(num_results);
		nanoflann::KNNResultSet<double> resultSet(num_results);
		resultSet.init(&ret_indexes[0], &out_dists_sqr[0] );
		mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

		// get the label
		int label = labels[ret_indexes[0]];

        // write label in file // label 0 is unknow need to add 1
        ofs << label+1 << endl;

        // iterate point id
        pt_id ++;
    }

    // close input and output files
    ifs.close();
    ofs.close();
}


void sem3d_estimate_attributes(char* filename, char* destination_filename, float normals_k){

	int K = int(normals_k);

	// load points
	vector<Voxel_center> v_centers;
	ifstream ifs (filename);
	std::string line;
	while(getline(ifs,line)){
		std::stringstream sstr(line);
		Voxel_center vox;
		sstr >> vox.x >> vox.y >> vox.z;
		v_centers.push_back(vox);
	}
	ifs.close();

	//////////////////
	// Z orientation 

	// create Eigen matrix
	Eigen::MatrixX3d points;
	Eigen::MatrixX3d normals;
	points.resize(v_centers.size(),3);
	for(size_t i=0; i<v_centers.size(); i++){
		points(i,0) = v_centers[i].x;
		points(i,1) = v_centers[i].y;
		points(i,2) = v_centers[i].z;
	}

	// normal estimation
	Eigen_Normal_Estimator ne(points, normals);
	ne.neighborhood_size = K;
	ne.estimate_normals();

	// orientation
	vector<int> z_orients(v_centers.size(),0);
    #pragma omp parallel for
	for(size_t pt_id=0; pt_id<z_orients.size(); pt_id++){

		Eigen::Vector3f normal(normals(pt_id,0), normals(pt_id,1), normals(pt_id,2));
		normal.normalize();
		double a = acos(fabs(float(normal.dot(Eigen::Vector3f(0,0,1)))));

		if(a != a){
			z_orients[pt_id] = 0;
			normals(pt_id, 0) = 0;
			normals(pt_id, 1) = 0;
			normals(pt_id, 2) = 1;
		}else{
			a /= M_PI /2.;
			a = (a > 1)? 1 : a;
			a = (a < 0)? 0 : a;
			z_orients[pt_id] = int(a*255);
		}
		if(z_orients[pt_id] < 0){
			cout << "Error " << a << endl;
			cout << normal[0] << " ";
			cout << normal[1] << " ";
			cout << normal[2] << endl;
		}
	}

	//////////////////
	// Noise estimation
	vector<int> noises(v_centers.size());
	for(size_t pt_id=0; pt_id<noises.size(); pt_id++){
		float noise = ne.noises[pt_id];
		if(noise != noise){
			noise = 0;
		}
		float threshold = 0.95;
		noise = (noise<threshold)? threshold: noise;
		noise = 1-noise;
		noise = noise/(1-threshold);
		noises[pt_id] = int(noise*255);
	}

	//////////////////
	// save data
	ofstream ofs (destination_filename);

	for(size_t pt_id=0; pt_id<z_orients.size(); pt_id++){
        ofs << v_centers[pt_id].x << " ";
		ofs << v_centers[pt_id].y << " ";
		ofs << v_centers[pt_id].z << " ";
		ofs << normals(pt_id, 0) << " ";
		ofs << normals(pt_id, 1) << " ";
		ofs << normals(pt_id, 2) << " ";
		ofs << z_orients[pt_id] << " ";
		ofs << noises[pt_id] << endl;
    }
    ofs.close();
}


#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h> 
#include <pcl/PCLPointCloud2.h>

Eigen::Vector3i label2color(int label){
	vector<Eigen::Vector3i> cols;
	cols.push_back(Eigen::Vector3i(0,0,0));
	cols.push_back(Eigen::Vector3i(192,192,192));
	cols.push_back(Eigen::Vector3i(0,255,0));
	cols.push_back(Eigen::Vector3i(38,214,64));
	cols.push_back(Eigen::Vector3i(247,247,0));
	cols.push_back(Eigen::Vector3i(255,3,0));
	cols.push_back(Eigen::Vector3i(122,0,255));
	cols.push_back(Eigen::Vector3i(0,255,255));
	cols.push_back(Eigen::Vector3i(255,110,206));
	return cols[label];
}


void sem3d_create_mesh(char* filename_rgb, 
						char* filename_composite,
						char* filename_mesh_rgb, 
						char* filename_mesh_composite,
						char* filename_mesh_label,
						char* filename_faces,
						int remove_multi_label_faces){

	vector<Eigen::Vector3d> points;
	vector<Eigen::Vector3i> rgbs;
	vector<Eigen::Vector3i> composites;
	vector<int> labels;

	// create the point cloud
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	ifstream ifs (filename_composite);
	std::string line;
	while(getline(ifs,line)){
		std::stringstream sstr(line);
		float x,y,z,nx,ny,nz;
		int z_orient, noise;
		sstr >> x >> y >> z >> nx >> ny >> nz >> z_orient >> noise;
		pcl::PointXYZRGBNormal pt;
		pt.x = x;
		pt.y = y; 
		pt.z = z;
		pt.normal_x = nx;
		pt.normal_y = ny;
		pt.normal_z = nz;
		cloud_with_normals->push_back(pt);

		Eigen::Vector3d point(x,y,z);
		Eigen::Vector3i composite(noise, z_orient, 0);
		points.push_back(point);
		composites.push_back(composite);

	}
	ifs.close();

	ifs.open(filename_rgb);
	while(getline(ifs,line)){
		std::stringstream sstr(line);
		float x,y,z;
		int r,g,b, label;
		if(remove_multi_label_faces){
			sstr >> x >> y >> z >> r >> g >> b >> label;
			Eigen::Vector3i rgb(r,g,b);
			rgbs.push_back(rgb);
			labels.push_back(label);
		}else{
			sstr >> x >> y >> z >> r >> g >> b;
			Eigen::Vector3i rgb(r,g,b);
			rgbs.push_back(rgb);
		}
	}
	ifs.close();

	cout << "Cloud size " << cloud_with_normals->size() << endl;

	pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
	tree->setInputCloud(cloud_with_normals);

	pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;
	pcl::PolygonMesh triangles;


    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius (2); // 2

    // Set typical values for the parameters
    gp3.setMu(2.5); // 2.5
    gp3.setMaximumNearestNeighbors (200); // 100
    gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
    gp3.setMinimumAngle(M_PI/18); // 10 degrees
    gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
    gp3.setNormalConsistency(true); // true


    // Get result
    gp3.setInputCloud (cloud_with_normals);
    gp3.setSearchMethod (tree);
    gp3.reconstruct (triangles);

    if(remove_multi_label_faces){

		// create the decimated triangles
        pcl::PolygonMesh triangles_decimated;
        triangles_decimated.header = triangles.header;
        triangles_decimated.cloud = triangles.cloud;

		// look at all trinagles
        for(size_t i=0; i<triangles.polygons.size(); i++){
            const int& v0 = labels[triangles.polygons[i].vertices[0]];
            const int& v1 = labels[triangles.polygons[i].vertices[1]];
            const int& v2 = labels[triangles.polygons[i].vertices[2]];
            if(v0==v1 && v0==v2){
                triangles_decimated.polygons.push_back(triangles.polygons[i]);
            }
        }

		// update triangles
        triangles = triangles_decimated;
	}


	// save mesh
	pcl::PCLPointCloud2 pc2;
	// save_mesh composite
	#pragma omp parallel for
	for(size_t pt_id=0; pt_id<cloud_with_normals->size(); pt_id++){
		cloud_with_normals->points[pt_id].r = rgbs[pt_id][0];
		cloud_with_normals->points[pt_id].g = rgbs[pt_id][1];
		cloud_with_normals->points[pt_id].b = rgbs[pt_id][2];
	}
	pcl::toPCLPointCloud2(*cloud_with_normals, pc2);
	triangles.cloud = pc2;
	pcl::io::savePolygonFile(filename_mesh_rgb,triangles);

	// save_mesh composite
	#pragma omp parallel for
	for(size_t pt_id=0; pt_id<cloud_with_normals->size(); pt_id++){
		cloud_with_normals->points[pt_id].r = composites[pt_id][0];
		cloud_with_normals->points[pt_id].g = composites[pt_id][1];
		cloud_with_normals->points[pt_id].b = composites[pt_id][2];
	}
	pcl::toPCLPointCloud2(*cloud_with_normals, pc2);
	triangles.cloud = pc2;
	pcl::io::savePolygonFile(filename_mesh_composite,triangles);

	// save mesh labels
	if(remove_multi_label_faces){
		#pragma omp parallel for
		for(size_t pt_id=0; pt_id<cloud_with_normals->size(); pt_id++){

			Eigen::Vector3i col = label2color(labels[pt_id]);
			cloud_with_normals->points[pt_id].r = col[0];
			cloud_with_normals->points[pt_id].g = col[1];
			cloud_with_normals->points[pt_id].b = col[2];
		}

		// update mesh colors
		pcl::toPCLPointCloud2(*cloud_with_normals, pc2);
		triangles.cloud = pc2;
		pcl::io::savePolygonFile(filename_mesh_label,triangles);
	}
		
	ofstream ofs(filename_faces);
	// look at all trinagles
	for(size_t i=0; i<triangles.polygons.size(); i++){
		ofs << triangles.polygons[i].vertices[0] << " ";
		ofs << triangles.polygons[i].vertices[1] << " ";
		ofs << triangles.polygons[i].vertices[2] << endl;
	}
	ofs.close();

}