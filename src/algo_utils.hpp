#include <iostream>
#include <vector>
#include "math.h"
#include <map>
#include <algorithm>
#include <cmath>
#include <random>
#pragma once

using namespace std;



template <typename T1>
void reinit(vector<vector<T1> > &container){

    for(int i=0;i<container.size(); i++){
        container[i].assign(container[i].size(), 0);
    }
}


// The following function is taken from the following question thread.
// https://stackoverflow.com/questions/20734774/random-array-generation-with-no-duplicates
void get_ranodm_indices(int *arr, int size, int seed)
{
    if (size > 1) 
    {
        int i = 0, j = 0, t = 0;
        srand(seed);
        
        for (i = 0; i < size - 1; i++) 
        {
          j = i + rand() / (RAND_MAX / (size - i) + 1);
          t = arr[j];
          arr[j] = arr[i];
          arr[i] = t;
        }
    }
}


template <typename T1, typename T2>
void init_centroids_randomly(vector<vector <T1> > &centroids, 
vector<vector <T1> > &dataset, T2 num_cluster, T2 seed){

    int i = 0, j = 0, size = dataset.size();
    int test_array[size];

    for (i = 0; i<size ; i++){
        test_array[i] = i;
    }
    
    get_ranodm_indices(test_array, size, seed);

    for(i=0; i<num_cluster; i++){  
        for(j=0; j <dataset[i].size(); j++){
            centroids[i][j] = dataset[test_array[i]][j];
        }   
    }
}

template <typename T1, typename T2>
void extract_data(vector<vector <T1> > &dataset, vector<vector <T1> > &extracted_data, 
T2 num_points, T2 num_cluster, T2 seed){


    int i = 0, j = 0, size = dataset.size();
    int test_array[size];

    for (i = 0; i<size ; i++){
        test_array[i] = i;
    }

    get_ranodm_indices(test_array, size, seed);
    
    for(i=0; i < num_points; i++){ 
        for(j=0; j<dataset[0].size(); j++){
            extracted_data[i][j] = dataset[test_array[i]][j];
        }   
        cout << endl;
    }
}



template <typename T1, typename T2>
void init_centroids_sequentially(vector<vector <T1> > &centroids, 
vector<vector <T1> > &dataset, T2 num_cluster){

    for(int i=0; i<num_cluster; i++){  
        for(int j=0; j<dataset[i].size(); j++){
                centroids[i][j] = dataset[i][j];
        }   
    }
}


inline float calc_sq_dist(const vector<float> &point, 
const vector<float> &center, long long int &dist_counter){
    
    float dist = 0.0;
    float temp = 0.0;
    
    for (int i=0; i < point.size(); i++){
        temp = point[i] - center[i];
        dist = dist + (temp*temp);
    }

    dist_counter += 1;
    return dist;
}


inline void calculate_distances(const vector<vector<float> > &dataset, 
vector<vector<float> > &centroids, vector<float> &dist_mat,
int num_clusters, vector<int> &assigned_clusters, vector<vector<float> > &cluster_info, 
long long int &dist_counter)

    {
    
    float temp1 = 0.0, temp2 = 0, shortestDist1 = 0.0;
    int i = 0, j = 0, fcenter = 0, scenter = 0;

    // Calculate the distance of points to nearest center
    for (i=0; i < dataset.size(); i++){
  
        temp1 = calc_sq_dist(dataset[i], centroids[0], dist_counter);
        shortestDist1 = std::numeric_limits<float>::max();

        for (j=0; j < centroids.size(); j++){ 
            
            temp1 = calc_sq_dist(dataset[i], centroids[j], dist_counter);
            
            if (temp1 < shortestDist1){
                shortestDist1 = temp1;
                fcenter = j;
            }
        }
        
        dist_mat[i] = shortestDist1;
        assigned_clusters[i] = fcenter;

        // Increase the size of the cluster
        cluster_info[fcenter][0] += 1;
        
        // Store the max so far
        if (shortestDist1 > cluster_info[fcenter][1])
            cluster_info[fcenter][1] = shortestDist1;

    }
}


inline void calculate_distances_dchw(const vector<vector<float> > &dataset, 
vector<vector<float> > &centroids, vector<float> &dist_mat,
int num_clusters, vector<int> &assigned_clusters, 
vector<vector<float> > &cluster_info, long long int &dist_counter)

    {
    
    float temp1 = 0.0, temp2 = 0, shortestDist1 = 0.0;
    int i = 0, j = 0, fcenter = 0, scenter = 0;

    // Calculate the distance of points to nearest center
    for (i=0; i < dataset.size(); i++){

        shortestDist1 = std::numeric_limits<float>::max();
  
        for (j=0; j < centroids.size(); j++){ 
            
            temp1 = calc_sq_dist(dataset[i], centroids[j], dist_counter);
            
            if (temp1 < shortestDist1){
                shortestDist1 = temp1;
                fcenter = j;
            }

        }
        
        dist_mat[i]= shortestDist1;
        assigned_clusters[i] = fcenter;

        // Increase the size of the cluster
        cluster_info[fcenter][0] += 1;
        
        // Store the max so far
        if (shortestDist1 > cluster_info[fcenter][1])
            cluster_info[fcenter][1] = shortestDist1;
    }
}


inline void update_centroids(vector<vector <float> > &dataset, 
vector<vector<float> > &new_centroids, vector<int> &assigned_clusters1, 
vector<vector<float> > &cluster_info, int numCols){

    int curr_center = 0, index = 0, k = 0, j =0;

    for (index=0; index<dataset.size(); index++){
        curr_center = assigned_clusters1[index];
        
        for (j = 0; j<numCols; j++){
            new_centroids[curr_center][j] += dataset[index][j];
        }
    }

    for(index=0; index<new_centroids.size();index++){
        
        k = cluster_info[index][0];

        if (k > 1){
            for (j = 0; j < new_centroids[index].size(); j++){
                new_centroids[index][j] = new_centroids[index][j]/k;
                }
            }
        
        else if (k < 1){
                new_centroids[index][j] = std::numeric_limits<float>::max();
        }

    }

}


inline bool check_convergence(vector<vector <float> > &new_centroids, 
vector<vector <float> > &centroids, int threshold){

    float temp_diff = 0, diff = 0;
    int i = 0, j = 0;

    if (new_centroids.size() == centroids.size()){
        
        for (i=0; i < new_centroids.size(); i++){
            for (j=0; j < new_centroids[i].size(); j++)
                temp_diff = new_centroids[i][j] - centroids[i][j];
                diff = diff + (temp_diff * temp_diff);
        }
        diff = sqrt(diff/new_centroids.size());
    }
    else
        return false;

    if (diff <= threshold)
        return true;
    return false;
}


bool check_status(vector<vector<float> > &centroids1, vector<vector<float> > &centroids2){

    for (int i = 0 ; i < centroids1.size(); i++){
        if(centroids1[i] != centroids2[i])
            return true;
    }
    return false;
}
