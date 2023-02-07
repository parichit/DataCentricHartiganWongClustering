#include <iostream>
#include <vector>
#include "math.h"
#include <algorithm>
#include "algo_utils.hpp"
#include <cmath>
#pragma once

using namespace std;


// void optrans(vector<vector<float> > &dataset, vector<vector<float> > &new_centroids, 
//             vector<vector<float> > dist_mat, vector<int> &assigned_clusters1, 
//             vector<int> &assigned_clusters2, vector<vector<float> > &cluster_info,  
//             vector<float> &an1, vector<float> &an2, int num_clusters,
//             long long int &dist_counter){

//     float temp1 = 0.0, temp2 = 0, shortestDist1 = 0.0, shortestDist2 = 0;
//     int i = 0, j = 0, fcenter = 0, scenter = 0, current_center;

//     // Calculate the distance of points to two nearest centers
//     for (i=0; i < dataset.size(); i++){

//         current_center = assigned_clusters1[i];
//         scenter = assigned_clusters2[i];
        
//         shortestDist1 = calc_sq_dist(dataset[i], new_centroids[current_center], dist_counter) * an1[current_center];
//         shortestDist2 = calc_sq_dist(dataset[i], new_centroids[scenter], dist_counter) * an2[scenter];

//         for (j=0; j < new_centroids.size(); j++){
            
//             temp1 = calc_sq_dist(dataset[i], new_centroids[j], dist_counter);
//             temp2 = temp1 * an2[j];
            
//             if (temp2 < shortestDist1){
//                 shortestDist1 = temp1;
//                 fcenter = j;
//             }

//             else if (temp2 < shortestDist2){
//                 shortestDist2 = temp1;
//                 scenter = j;
//             }

//         }

//         if (fcenter != current_center){
//             assigned_clusters1[i] = fcenter;

//             cluster_info[current_center][0] -= 1;
//             cluster_info[fcenter][0] += 1;
            
//             dist_mat[i][0] = shortestDist1;
//             dist_mat[i][1] = shortestDist2;
//             assigned_clusters2[i] = scenter;
//         }
        
//         // Store the max so far
//         if (shortestDist1 > cluster_info[fcenter][1])
//             cluster_info[fcenter][1] = shortestDist1;
//     }

//     for (i = 0; i < num_clusters; i++){
//         an1[i] = cluster_info[i][0]/(cluster_info[i][0]-1);
//         an2[i] = cluster_info[i][0]/(cluster_info[i][0]+1);
//     }         

// }


// void quicktrans(vector<vector<float> > &dataset, vector<vector<float> > &new_centroids, 
//             vector<vector<float> > dist_mat, vector<int> &assigned_clusters1, 
//             vector<int> &assigned_clusters2, vector<vector<float> > &cluster_info,  
//             vector<float> &an1, vector<float> &an2, int num_clusters,
//             long long int &dist_counter){

    
//     float temp1 = 0.0, temp2 = 0, shortestDist1 = 0.0, shortestDist2 = 0;
//     int i = 0, j = 0, fcenter = 0, scenter = 0, current_center;


//     // Calculate the distance of points to two nearest centers
//     for (i=0; i < dataset.size(); i++){

//         fcenter = assigned_clusters1[i];
//         scenter = assigned_clusters2[i];
//         shortestDist1 = calc_sq_dist(dataset[i], new_centroids[fcenter], dist_counter) * an1[fcenter];
//         shortestDist2 = calc_sq_dist(dataset[i], new_centroids[scenter], dist_counter) * an2[scenter];
//         dist_mat[i][0] = shortestDist1;
        
//         if (shortestDist2 < shortestDist1){
//             current_center =  fcenter;
//             fcenter = scenter;
//             scenter = current_center;

//             dist_mat[i][0] = shortestDist2;
//             dist_mat[i][1] = shortestDist1;

//             assigned_clusters1[i] = fcenter;
//             assigned_clusters2[i] = scenter;

//             // Increase the size of the cluster
//             cluster_info[scenter][0] -= 1;
//             cluster_info[fcenter][0] += 1;

//             // Store the max so far
//             if (shortestDist2 > cluster_info[fcenter][1])
//                 cluster_info[fcenter][1] = shortestDist2;
//         }

//         else{
//             dist_mat[i][0] = shortestDist1;
//             dist_mat[i][1] = shortestDist2;

//             // Store the max so far
//             if (shortestDist1 > cluster_info[fcenter][1])
//                 cluster_info[fcenter][1] = shortestDist1;
//          }  

//     }

//     for (i = 0; i < num_clusters; i++){
//         an1[i] = cluster_info[i][0]/(cluster_info[i][0]-1);
//         an2[i] = cluster_info[i][0]/(cluster_info[i][0]+1);
//     }  
// }


inline void reassign_point(vector<float> &point, int index, vector<float> &dist_matrix, 
vector<vector<float> > &new_centroids, vector<int> &assigned_clusters, int curr_cluster, 
vector<vector<float> > &cluster_info, 
vector<float> &an1, vector<float> &an2, long long int &dist_counter){

float temp = 0, my_sse = 0, ot_sse = 0, 
shortest_sse = std::numeric_limits<float>::max(), shortest_dist = 0;
int new_cluster = 0;

my_sse = calc_sq_dist(point, new_centroids[curr_cluster], dist_counter);

if (cluster_info[curr_cluster][0] > 1){
    my_sse *= an1[curr_cluster];
}


for (int ot_cluster = 0 ; ot_cluster < new_centroids.size(); ot_cluster++){
    if(curr_cluster != ot_cluster){
        ot_sse = (calc_sq_dist(point, new_centroids[ot_cluster], dist_counter) * an2[ot_cluster]);
        if (ot_sse < shortest_sse){
            shortest_sse = ot_sse;
            shortest_dist = ot_sse/an2[ot_cluster];
            new_cluster = ot_cluster;
        }
    }
}

if (shortest_sse < my_sse){
    assigned_clusters[index] = new_cluster;
    // dist_matrix[index] = shortest_dist;
    cluster_info[curr_cluster][0] = cluster_info[curr_cluster][0] - 1;
    cluster_info[new_cluster][0] = cluster_info[new_cluster][0] + 1;
    }

}