#include <iostream>
#include <vector>
#include "math.h"
#include <map>
#include <algorithm>
#include "algo_utils.hpp"
#include "misc_utils.hpp"
#include <numeric>
#pragma once

using namespace std;

class dckm_utils{

    template <typename Tfloat>
    inline void find_midpoints(const vector<Tfloat> &center1, const vector<Tfloat> &center2, 
    vector<vector<vector<Tfloat> > > &midpoint, vector<vector<vector<Tfloat> > > &affine, 
    int &curr_center, int &ot_cen);

    template <typename Tfloat>
    inline bool find_context_direction(const vector<Tfloat> &actual_point, 
    const vector<Tfloat> &centroid_vector, const vector<Tfloat> &midpoint, 
    Tfloat &vec_sum);

    template <typename TD, typename TI>
    void restore_radius(vector<vector <TD> > &dist_matrix,
    vector<TI> &assigned_clusters, 
    vector<vector <TD> > &cluster_info);

    template <typename TD, typename TI>
    void find_neighbors(vector<vector <TD> > &centroids, 
    vector<vector <TD> > &center_dist_mat, vector<vector <TD> > &cluster_info, 
    vector<vector<TI> > &neighbors, vector<vector<vector <TD> > > &mid_points, 
    vector<vector<vector <TD> > > &affine_vectors,  vector<TD> &temp, 
    vector<vector<TD> > &temp_master, vector<TD> &temp_midpoint,
    vector<TD> &temp_affine, vector<vector<TD> > &midpoint_holder, 
    vector<vector<TD> > &affine_holder, unsigned long long &dist_counter);

    template <typename Tfloat, typename Tint>
    inline void determine_data_expression(vector<vector<Tfloat> > &dataset, 
    vector<vector <Tfloat> > &centroids, vector<vector<Tfloat> > &cluster_info, 
    vector<vector <Tfloat> > &center_dist_mat, vector<vector<Tfloat> > &dist_mat,
    vector<Tint> &assigned_clusters, 
    vector<vector<Tint> > &neighbors,
    vector<vector<vector <Tfloat> > > &affine_vectors, 
    vector<vector<vector <Tfloat> > > &mid_points, 
    unsigned long long &dist_counter, vector<Tint> &temp, Tint &my_cluster, 
    Tint &i, Tint &j, Tfloat &temp_result);

};


template <typename Tfloat>
inline void find_midpoints(const vector<Tfloat> &center1, const vector<Tfloat> &center2, 
vector<vector<vector<Tfloat> > > &midpoint, vector<vector<vector<Tfloat> > > &affine, 
int &curr_center, int &ot_cen){

    for (int i=0; i<center1.size(); i++){
        midpoint[curr_center][ot_cen][i] = (center1[i] + center2[i])/2;
        affine[curr_center][ot_cen][i] = center2[i] - midpoint[curr_center][ot_cen][i];
    }
}


template <typename Tfloat>
inline bool find_context_direction(const vector<Tfloat> &actual_point, 
const vector<Tfloat> &centroid_vector, const vector<Tfloat> &midpoint, 
Tfloat &vec_sum){

    vec_sum = 0.0;
    
    for (int i=0; i<midpoint.size(); i++){
        vec_sum = vec_sum + ((actual_point[i] - midpoint[i]) * centroid_vector[i]);
    }

    if (vec_sum>0)
        return true;

    return false;
}


template <typename TD, typename TI>
inline void restore_radius(vector<vector <TD> > &dist_matrix,
vector<TI> &assigned_clusters, 
vector<vector <TD> > &cluster_info){

    for (int i=0; i<cluster_info.size(); i++){
        for (int j=0; j< assigned_clusters.size(); j++){
                if (assigned_clusters[j] == i){
                    if(dist_matrix[j][i] > cluster_info[i][1]){
                            cluster_info[i][1] = dist_matrix[j][i];
                    }
                }
            }
        }
    }



template <typename TD, typename TI>
inline void find_neighbors(vector<vector <TD> > &centroids, 
vector<vector <TD> > &center_dist_mat, vector<vector <TD> > &cluster_info, 
vector<vector<TI> > &neighbors, vector<vector<vector <TD> > > &mid_points, 
vector<vector<vector <TD> > > &affine_vectors, vector<TD> &temp, 
vector<vector<TD> > &temp_master, vector<TD> &temp_midpoint,
vector<TD> &temp_affine, vector<vector<TD> > &midpoint_holder, 
vector<vector<TD> > &affine_holder, unsigned long long &dist_counter){

    TD dist = 0;
    TD radius = 0;
    algorithm_utils alg_utils;
    vector<TI> temp1;

    int curr_center = 0, ot_center = 0, cnt = 0;

    // Calculate inter-centroid distances
    for(curr_center=0; curr_center<centroids.size(); curr_center++){
        
        radius = cluster_info[curr_center][1];
        cnt = 0;
        
        for (ot_center=0; ot_center<centroids.size(); 
        ot_center++){    
            
            // Do only k calculations, save so many :)
            if (curr_center < ot_center){
                dist = alg_utils.calc_euclidean(centroids[curr_center], centroids[ot_center], dist_counter);
                center_dist_mat[curr_center][ot_center] = dist/2;
                center_dist_mat[ot_center][curr_center] = center_dist_mat[curr_center][ot_center];
            }

            // Start neighbor finding
            if ((curr_center != ot_center) && 
            (center_dist_mat[curr_center][ot_center] < radius)){

                // Create an object of neighbor holder structure
                // and populate the fields inside it.
                temp[0] = center_dist_mat[curr_center][ot_center];
                temp[1] = ot_center;
                temp[2] = cnt;
                temp_master.push_back(temp);
           
                // Get the mid-point coordinates for this pair of centroids
                find_midpoints(centroids[curr_center], centroids[ot_center], mid_points, affine_vectors, curr_center, ot_center);
                cnt++;
            }
        }   

            if (cnt>1){
                
                sort(temp_master.begin(), temp_master.end(), [](const std::vector<TD>& a, const std::vector<TD>& b) {
                    return a[0] < b[0];});

                for(int i = 0; i<temp_master.size(); i++){
                    temp1.push_back(trunc(temp_master[i][1]));
                }

                neighbors[curr_center] = temp1;
            }

            else if (cnt == 1){
                temp1.push_back(temp_master[0][1]);
                neighbors[curr_center] = temp1;
            }
 
            else if(cnt == 0){
                temp1.push_back(-100);
                neighbors[curr_center] = temp1;
            }

            cluster_info[curr_center][2] = cnt;
            temp_master.clear();
            temp1.clear();
    }
}


template <typename Tfloat, typename Tint>
inline void determine_data_expression(vector<vector<Tfloat> > &dataset, 
vector<vector <Tfloat> > &centroids, vector<vector<Tfloat> > &cluster_info, 
vector<vector <Tfloat> > &center_dist_mat, vector<vector<Tfloat> > &dist_mat,
vector<Tint> &assigned_clusters, 
vector<vector<Tint> > &neighbors,
vector<vector<vector <Tfloat> > > &affine_vectors, 
vector<vector<vector <Tfloat> > > &mid_points, 
unsigned long long &dist_counter, vector<Tint> &temp, Tint &my_cluster, 
Tint &i, Tint &j, Tfloat &temp_result){
    

    algorithm_utils alg_utils;

    for (i = 0; i < assigned_clusters.size(); i++){

        my_cluster = assigned_clusters[i];

        if (cluster_info[my_cluster][2] > 0){
            
            temp_result = alg_utils.calc_euclidean(dataset[i], centroids[my_cluster], dist_counter);
            if(temp_result < center_dist_mat[my_cluster][neighbors[my_cluster][0]]){
                continue;    
            }

            dist_mat[i][my_cluster] = temp_result;


            for (j=0; j<neighbors[my_cluster].size(); j++){ 
                
                if (find_context_direction(dataset[i], affine_vectors[my_cluster][neighbors[my_cluster][j]], 
                mid_points[my_cluster][neighbors[my_cluster][j]], temp_result)){

                    temp_result = alg_utils.calc_euclidean(dataset[i], centroids[neighbors[my_cluster][j]], dist_counter);
                    dist_mat[i][neighbors[my_cluster][j]] = temp_result;

                    if(temp_result > cluster_info[neighbors[my_cluster][j]][1])
                        cluster_info[neighbors[my_cluster][j]][1] = temp_result;

                    if(temp_result < dist_mat[i][assigned_clusters[i]]){
                        cluster_info[assigned_clusters[i]][0] -= 1;
                        assigned_clusters[i] = neighbors[my_cluster][j];
                        cluster_info[neighbors[my_cluster][j]][0] += 1;
                    } 

                }
            }
        } 
    }      
}
            
