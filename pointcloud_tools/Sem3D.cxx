#include "Sem3D.h"

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
int color2label(Eigen::Vector3i col){
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
    for(size_t i=0; i<cols.size(); i++){
        if(col == cols[i]){
            return i;
        }
    }
	return 0;
}

Sem3D::Sem3D(){
    pc = PointCloudPtr(new PointCloud);
    srand(time(NULL));
}

Sem3D::~Sem3D(){
}

void Sem3D::set_voxel_size(float vox_size){
    voxel_size = vox_size;
}

void Sem3D::load_Sem3D(char* filename){

    //cout << "Rectange load Sem3D" << endl;

    // clear the point cloud
	pc->clear();

    // open the semantic 3D file
	std::ifstream ifs(filename);
	std::string line;
	int pt_id =0;


	std::map<Eigen::Vector3i, Voxel_center, Vector3icomp> voxels;
	while(getline(ifs,line)){
		pt_id++;
		if((pt_id+1)%1000000==0){
			cout << (pt_id+1)/1000000 << " M" << endl;
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

	// build the new point cloud
	pc->resize(voxels.size());
	//intensities.resize(voxels.size());

	pt_id = 0;
	for(std::map<Eigen::Vector3i, Voxel_center>::iterator it=voxels.begin(); it != voxels.end(); it++){
		pc->points[pt_id].x = it->second.x;
		pc->points[pt_id].y = it->second.y;
		pc->points[pt_id].z = it->second.z;
		pc->points[pt_id].r = it->second.r;
		pc->points[pt_id].g = it->second.g;
		pc->points[pt_id].b = it->second.b;
		//intensities[pt_id] = it->second.intensity;
		pt_id++;
	}

	ifs.close();
}

void Sem3D::load_Sem3D_labels(char* filename, char* labels_filename){
    //cout << "Rectange load Sem3D with labels" << endl;

	pc->clear();
	labels.clear();

	std::ifstream ifs(filename);
	std::ifstream ifs_labels(labels_filename);
	std::string line;
	std::string line_labels;
	int pt_id =0;

	std::map<Eigen::Vector3i, Voxel_center, Vector3icomp> voxels;
	while(getline(ifs,line)){
		pt_id++;
		if((pt_id+1)%1000000==0){
			cout << (pt_id+1)/1000000 << " M" << endl;
		}
		getline(ifs_labels, line_labels);

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
 
	// build the new point cloud
	pc->resize(voxels.size());
	labels.resize(voxels.size());

	pt_id = 0;
	for(std::map<Eigen::Vector3i, Voxel_center>::iterator it=voxels.begin(); it != voxels.end(); it++){
		pc->points[pt_id].x = it->second.x;
		pc->points[pt_id].y = it->second.y;
		pc->points[pt_id].z = it->second.z;
		pc->points[pt_id].r = it->second.r;
		pc->points[pt_id].g = it->second.g;
		pc->points[pt_id].b = it->second.b;
		labels[pt_id] = it->second.label;
		pt_id++;
	}

	ifs.close();
}



void Sem3D::load_ply_labels(char*filename){
    PointCloudPtr pc_temp(new PointCloud);
    pcl::io::loadPLYFile(filename, *pc_temp);
    labels.resize(pc_temp->size());
    for(size_t pt_id=0; pt_id<pc_temp->size(); pt_id++){
        labels[pt_id] = color2label(pc_temp->points[pt_id].getRGBVector3i());
    }
}

void Sem3D::save_ply_labels(char* filename){

    vector<Eigen::Vector3i> labels_colors(pc->size());
    #pragma omp parallel for
    for(size_t pt_id=0; pt_id<pc->size(); pt_id++){
        labels_colors[pt_id][0] = int(pc->points[pt_id].r);
        labels_colors[pt_id][1] = int(pc->points[pt_id].g);
        labels_colors[pt_id][2] = int(pc->points[pt_id].b);

        Eigen::Vector3i col = label2color(labels[pt_id]);

        pc->points[pt_id].r = col[0];
        pc->points[pt_id].g = col[1];
        pc->points[pt_id].b = col[2];
    }
    save_ply(filename);

    // give colors back to point cloud
    #pragma omp parallel for
    for(size_t pt_id=0; pt_id<pc->size(); pt_id++){
        pc->points[pt_id].r = labels_colors[pt_id][0];
        pc->points[pt_id].g = labels_colors[pt_id][1];
        pc->points[pt_id].b = labels_colors[pt_id][2];
    }
}

void Sem3D::save_mesh_labels(char* filename){

    // create rgb labels colors
    vector<Eigen::Vector3i> labels_colors(pc->size());
    #pragma omp parallel for
	for(size_t pt_id=0; pt_id<pc->size(); pt_id++){
        labels_colors[pt_id][0] = int(pc->points[pt_id].r);
        labels_colors[pt_id][1] = int(pc->points[pt_id].g);
        labels_colors[pt_id][2] = int(pc->points[pt_id].b);

        Eigen::Vector3i col = label2color(labels[pt_id]);

        pc->points[pt_id].r = col[0];
        pc->points[pt_id].g = col[1];
        pc->points[pt_id].b = col[2];
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
        pc->points[pt_id].r = labels_colors[pt_id][0];
        pc->points[pt_id].g = labels_colors[pt_id][1];
        pc->points[pt_id].b = labels_colors[pt_id][2];
	}
}

void Sem3D::get_labelsColors(int* array, int m, int n)
{
    int i; //, j ;
    int index = 0 ;
    for (i = 0; i < m; i++) {
        Eigen::Vector3i col = label2color(labels[i]);
        array[index] = col[0];
        array[index+1] = col[1];
        array[index+2] = col[2];
        index+=3;
    }
    return ;
}



void Sem3D::remove_unlabeled_points(){

    bool use_noises = (noises.size()==pc->size());
    bool use_z_orient = (z_orients.size() == pc->size());

    int current_pos = 0;
    for(size_t pt_id=0; pt_id<pc->size(); pt_id++){
        if(labels[pt_id]==0){
            continue;
        }
        pc->points[current_pos] = pc->points[pt_id];
        labels[current_pos] = labels[pt_id];
        if(use_noises)
            noises[current_pos] = noises[pt_id];
        if(use_z_orient)
            z_orients[current_pos] = z_orients[current_pos];
        current_pos++;
    }
    labels.resize(current_pos);
    pc->resize(current_pos);
    if(use_noises)
        noises.resize(current_pos);
    if(use_z_orient)
        z_orients.resize(current_pos);

}

void Sem3D::mesh_to_label_file_no_labels(char* mesh_filename,
		char* sem3d_cloud_txt,
		char* output_results){
	// load the mesh
	PointCloudPtr pc_temp(new PointCloud);
	pcl::io::loadPLYFile(mesh_filename, *pc_temp);

	// removing unlabeled points
	int nbr_pts = 0;
	for(size_t pt_id=0; pt_id<pc_temp->size(); pt_id++){
		if(pc_temp->points[pt_id].getRGBVector3i() == Eigen::Vector3i(0,0,0)) continue;
		nbr_pts++;
	}
	pc->resize(nbr_pts);
	int pos=0;
	for(size_t pt_id=0; pt_id<pc_temp->size(); pt_id++){
		if(pc_temp->points[pt_id].getRGBVector3i() == Eigen::Vector3i(0,0,0)) continue;
		pc->points[pos] = pc_temp->points[pt_id];
		pos++;
	}
	cout << pc->size() << endl;

	// build the KDtree
	pcl::KdTreeFLANN<Point> kdtree;
	kdtree.setInputCloud(pc);

    // create the voxel map
    std::map<Eigen::Vector3i, int, Vector3icomp> voxels;
    for(size_t pt_id=0; pt_id<pc->size(); pt_id++){
        int x_id = std::floor(pc->points[pt_id].x/voxel_size) + 0.5; // + 0.5, centre du voxel (k1*res, k2*res)
		int y_id = std::floor(pc->points[pt_id].y/voxel_size) + 0.5;
		int z_id = std::floor(pc->points[pt_id].z/voxel_size) + 0.5;
        Eigen::Vector3i vox(x_id, y_id, z_id);
        voxels[vox] = color2label(pc->points[pt_id].getRGBVector3i());
    }


	// iterate over the file
	ofstream ofs(output_results);
	ifstream ifs(sem3d_cloud_txt);
    string line;
    int pt_id = 0;
    while (std::getline(ifs, line))
    {
        if((pt_id+1) % 10000==0){
            cout << pt_id+1 << endl;
        }

        std::istringstream iss(line);
        float x,y,z;
        iss >> x >> y >> z;

        int label = 0;

        // test if point in the map
        int x_id = std::floor(x/voxel_size) + 0.5; // + 0.5, centre du voxel (k1*res, k2*res)
		int y_id = std::floor(y/voxel_size) + 0.5;
		int z_id = std::floor(z/voxel_size) + 0.5;
        Eigen::Vector3i vox(x_id, y_id, z_id);
        if(voxels.count(vox)>0){
            label = voxels[vox];
        }else{
            // operate a nearest neighbor search
            Point pt;
            pt.x = x;
            pt.y = y;
            pt.z = z;
            int K = 1;
            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);
            kdtree.nearestKSearch(pt, K, pointIdxNKNSearch, pointNKNSquaredDistance);
            label = color2label(pc->points[pointIdxNKNSearch[0]].getRGBVector3i());

        }

        // write label in file
        ofs << label << endl;

        // iterate point id
        pt_id ++;
    }

    // close input and output files
    ifs.close();
    ofs.close();
}
