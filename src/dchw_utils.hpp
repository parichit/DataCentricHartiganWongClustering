#include <iostream>
#include <vector>
#include "math.h"
#include <algorithm>
#include "algo_utils.hpp"
#include <cmath>
#pragma once

using namespace std;


inline bool find_context_direction(const vector<float> &actual_point, 
const vector<float> &centroid_vector, const vector<float> &midpoint, 
float &vec_sum){

    vec_sum = 0.0;
    
    for (int i=0; i<midpoint.size(); i++){
        vec_sum = vec_sum + ((actual_point[i] - midpoint[i]) * centroid_vector[i]);
    }

    if (vec_sum>0)
        return true;

    return false;
}



inline void find_cutoff_points(vector<float> &curr_cluster, vector<float> &ot_center, 
vector<vector<vector <float> > > &cutoff_points, vector<vector<vector <float> > > &affine_vectors, 
vector<vector<float> > &cluster_info, float &neighbor_dist,
 int &curr_center_index, int &ot_center_index, vector<float> &an1, vector<float> &an2, 
 float &scale_fac){

    vector<float> direc_vec(curr_cluster.size(), 0);
    int i = 0;
    float acc_sum = 0;
    bool flag = false;
    scale_fac = 0;

    // 1. Find direction vector
    for (i = 0 ; i < curr_cluster.size(); i++){
        direc_vec[i] = ot_center[i] - curr_cluster[i];
        acc_sum += direc_vec[i]*direc_vec[i];
    }

    // 2. Calculate the unit vector
    acc_sum = sqrt(acc_sum);
    for (i = 0 ; i < curr_cluster.size(); i++)
        direc_vec[i] = direc_vec[i]/acc_sum ;


    // 3. Get the scale factor
    if (cluster_info[ot_center_index][0] < cluster_info[curr_center_index][0]){
        scale_fac = (cluster_info[ot_center_index][0]/cluster_info[curr_center_index][0])*100 ;
    }
    else{
        scale_fac = (cluster_info[curr_center_index][0]/cluster_info[ot_center_index][0])*100 ;
        flag = true;
    }

    if (scale_fac >= neighbor_dist){
        
        if (flag == true){
            scale_fac = (cluster_info[ot_center_index][0]/cluster_info[curr_center_index][0]) + (an1[curr_center_index]/an2[ot_center_index]);
        }
        else{
            scale_fac = (cluster_info[curr_center_index][0]/cluster_info[ot_center_index][0]) + (an1[curr_center_index]/an2[ot_center_index]);   
        }
        scale_fac = neighbor_dist - scale_fac;
    }
    
    // 4. Scale the unit vector and get the cut-off coordinates
    for (i = 0 ; i < curr_cluster.size(); i++){
        // cutoff_points[curr_center_index][ot_center_index][i] = (curr_cluster[i] + ot_center[i])/2;
        cutoff_points[curr_center_index][ot_center_index][i] = curr_cluster[i] + (scale_fac * direc_vec[i]);
        affine_vectors[curr_center_index][ot_center_index][i] = ot_center[i] - (cutoff_points[curr_center_index][ot_center_index][i]);
    }

    // if (curr_center_index ==0){
    //     print_vector(direc_vec, direc_vec.size(), "Direc vector");
    //     cout << scale_fac << endl;
    //     print_2d_vector(cutoff_points[curr_center_index], cutoff_points[curr_center_index].size(), "Cutoff");
    // }
}


inline void find_neighbors(vector<vector <float> > &centroids, 
vector<vector <float> > &center_dist_mat, vector<vector <float> > &cluster_info, 
vector<vector<int> > &neighbors, vector<vector<vector <float> > > &cutoff_points, 
vector<vector<vector <float> > > &affine_vectors, vector<float> &an1, vector<float> &an2,
vector<float> &cluster_safe){

    float dist = 0;
    float radius = 0, scale_fac = 0;
    
    // Clear previous allocations
    // reinit(neighbors);
    vector<int> temp;

    int curr_center = 0, ot_center = 0, cnt = 0;
    float limit = 0;

    // Calculate inter-centroid distances
    for(curr_center=0; curr_center<centroids.size(); curr_center++){
        
        radius = cluster_info[curr_center][1];
        cnt = 0;
        
        for(ot_center=0; ot_center<centroids.size(); 
        ot_center++){ 

            limit = std::numeric_limits<float>::max();   
            
            // Do only k calculations, save so many :)
            if (curr_center < ot_center){
                dist = calc_euclidean(centroids[curr_center], centroids[ot_center]);
                center_dist_mat[curr_center][ot_center] = dist;
                center_dist_mat[ot_center][curr_center] = center_dist_mat[curr_center][ot_center];
            }

            // Start neighbor finding
            if ((curr_center != ot_center) && 
            (center_dist_mat[curr_center][ot_center]/2 < radius)){

                // Create an object of neighbor holder structure
                // and populate the fields inside it.
                // temp[0] = center_dist_mat[curr_center][ot_center];
                // temp[1] = ot_center;
                // temp[2] = cnt;
                // temp_master.push_back(temp);
                temp.push_back(ot_center);
                // neighbors[curr_center].push_back(ot_center);
           
                // Get the cut-off coordinates and affine vector for this pair of centroids
                // find_cutoff_points(centroids[curr_center], centroids[ot_center], cutoff_points, affine_vectors, 
                // cluster_info, center_dist_mat[curr_center][ot_center], curr_center, ot_center, an1, an2, scale_fac);
                cnt++;

                // if(scale_fac < limit){
                //     limit = scale_fac;
                //     cluster_safe[curr_center] = scale_fac;
                // }

            }
        }
        
        if (cnt > 0){
            neighbors[curr_center] = temp;
        }
        
        cluster_info[curr_center][2] = cnt;   
        
        temp.clear();
    }
}




inline void update_cluster_info(vector<int> &assigned_clusters, 
vector<vector<float> >  &cluster_info, vector<float> &dist_matrix,
vector<float> &an1, vector<float> &an2, int num_clusters){

    int curr_center = 0, i = 0;

    for (int i = 0 ; i < assigned_clusters.size() ; i++){

        curr_center = assigned_clusters[i];

        if (cluster_info[curr_center][1] < dist_matrix[i]){
            cluster_info[curr_center][1] = dist_matrix[i];
        }
    }

    for (i = 0; i < num_clusters; i++){
        
        if (cluster_info[i][0] > 1){
            an1[i] = cluster_info[i][0]/(cluster_info[i][0]-1);
            cluster_info[i][1] *= an1[i];
        }
        an2[i] = cluster_info[i][0]/(cluster_info[i][0]+1);
        // cluster_info[i][1] *= an1[i];
    }

}