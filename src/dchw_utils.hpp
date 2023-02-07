#include <iostream>
#include <vector>
#include "math.h"
#include <algorithm>
#include "algo_utils.hpp"
#include <cmath>
#pragma once

using namespace std;



inline void get_he_data(vector<vector<float> > &dataset, vector<float> &dist_mat, vector<float> &centroid,
vector<float> &an1, int clus, vector<int> &assigned_clusters, int &num_clusters, 
vector<vector <float> > &cluster_info, vector<vector<float> > &he_data, long long &dist_counter){

    int curr_cluster = 0, i = 0;
    vector<float> temp1(2, 0);
    float temp = 0;

    he_data.clear();

    // Find HE data
    for (i = 0 ; i < dist_mat.size(); i++){
        
        curr_cluster = assigned_clusters[i];
        
        if (curr_cluster == clus){

            temp = calc_sq_dist(dataset[i], centroid, dist_counter);
        
            if(temp > cluster_info[curr_cluster][1]/2){
                temp1[0] = i;
                temp1[1] = temp;
                he_data.push_back(temp1);

            }
        }
    
    }

}


inline void reassign_he_data(vector<vector<float> > &dataset, vector<vector<float> > &new_centroids, 
                        vector<vector<float> > &cluster_info, vector<vector<float> > &he_data, 
                        vector<int> &assigned_clusters, int curr_cluster, vector<float> &dist_matrix,
                        vector<float> &an1, vector<float> &an2, long long int &dist_counter){

    int i =0, j =0, ot_cluster = 0, point  = 0, new_center = 0;
    float temp = 0, temp1 = 0, my_sse = 0, 
    shortest_sse = std::numeric_limits<float>::max(), shortest_dist = 0;

    if (cluster_info[curr_cluster][0] > 1){

        for(i=0; i<he_data.size(); i++){
            he_data[i][1] *= an1[curr_cluster];
        }
    }
    

    for (i = 0 ; i < he_data.size();  i++){

        point = he_data[i][0];
        my_sse = he_data[i][1];
        
        for(ot_cluster = 0; ot_cluster < new_centroids.size(); ot_cluster++){

            if (curr_cluster != ot_cluster){

                // Calculate SSE
                temp = calc_sq_dist(dataset[point], new_centroids[ot_cluster], dist_counter);
                // temp1 = temp * (cluster_info[ot_cluster][0]/(cluster_info[ot_cluster][0]+1));
                temp1 = temp * an2[ot_cluster];

                if (temp1 < shortest_sse){

                    shortest_sse = temp1;
                    shortest_dist = temp;
                    new_center = ot_cluster;

                    // dist_matrix[point] = temp;
                    // assigned_clusters[point] = ot_cluster;

                    // cluster_info[curr_cluster][0] -= 1;
                    // cluster_info[ot_cluster][0] += 1;

                }

            }

        }

        if (shortest_sse < my_sse){
            assigned_clusters[point] = new_center;
            dist_matrix[point] = shortest_dist;
            cluster_info[curr_cluster][0] = cluster_info[curr_cluster][0] - 1;
            cluster_info[new_center][0] = cluster_info[curr_cluster][0] + 1;
        }

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

