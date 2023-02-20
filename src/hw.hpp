#include <iostream>
#include <vector>
#include "misc_utils.hpp"
#include "algo_utils.hpp"
#include "HW_utils.hpp"
#include <chrono>

using namespace std;


inline output_data hw_kmeans(vector<vector <float> > &dataset, int num_clusters, 
float threshold, int num_iterations, int numCols, int time_limit){

    
    
    vector<vector<float> > centroids(num_clusters, vector<float>(numCols, 0));
    vector<vector<float> > new_centroids(num_clusters, vector<float>(numCols, 0));
    vector<float> dist_matrix(dataset.size());
    
    vector<vector<float> > cluster_info(num_clusters, vector<float>(2, 0));  
    vector<float> an1(num_clusters);  
    vector<float> an2(num_clusters);  

    vector<int> assigned_clusters(dataset.size(), 0);
 
    // Create objects
    print_utils pu;

    long long int dist_counter = 0;
    int loop_counter = 0, curr_cluster = 0, ot_cluster = 0;
    bool centroid_status = false;

    output_data result;

    // Start time counter 
    // auto start = std::chrono::high_resolution_clock::now();
    
    // Initialize centroids
    init_centroids_sequentially(centroids, dataset, num_clusters);

    calculate_distances(dataset, centroids, dist_matrix, 
    num_clusters, assigned_clusters, cluster_info, 
    dist_counter);

    // Check for empty clusters and return
    for (int i=0; i<num_clusters; i++){
    
        if(cluster_info[i][0] == 0){
            cout << "Empty cluster found after intialization, safe exiting" << endl;
            result.loop_counter = 0;
            result.num_dist = 0;
            result.runtime = 0;
            result.timeout = false;
            return result;
        }
    }
    
    // Update an1 and an2
    for (int i = 0; i < num_clusters; i++){
        if (cluster_info[i][0] > 1)
            an1[i] = cluster_info[i][0]/(cluster_info[i][0]-1);  
        
        an2[i] = cluster_info[i][0]/(cluster_info[i][0]+1);
    }

    // Calculate new centroids
    update_centroids(dataset, new_centroids, assigned_clusters, 
    cluster_info, numCols);

    // print_2d_vector(centroids, centroids.size(), "Initial");


    while (loop_counter < num_iterations){

         // Check Convergence
        if (check_status(centroids, new_centroids)){

            loop_counter++;
            
            centroid_status =  true;

            // cout << loop_counter << endl;
            // print_2d_vector(new_centroids, new_centroids.size(), "Inside loop");

            for (int i =0 ; i< dataset.size(); i++){
                curr_cluster = assigned_clusters[i];
                if (cluster_info[curr_cluster][0] > 1){
                    reassign_point(dataset[i], i, dist_matrix, new_centroids, assigned_clusters, curr_cluster, 
                    cluster_info, an1, an2, dist_counter);
                }
            }
            
            for (int i = 0; i < num_clusters; i++){
                if (cluster_info[i][0] > 1)
                    an1[i] = cluster_info[i][0]/(cluster_info[i][0]-1);
                an2[i] = cluster_info[i][0]/(cluster_info[i][0]+1);
                }
            
            }
      
        else{
            centroid_status =  false; 
            }
            
        if (centroid_status == false){
            cout << "HW convergence at iteration: " <<  loop_counter << endl;
            break;
        }

        // Check for empty clusters and return
        for (int i=0; i<num_clusters; i++){
    
            if(cluster_info[i][0] == 0){
                cout << "Empty cluster found during iter, safe exiting" << endl;
                result.loop_counter = 0;
                result.num_dist = 0;
                result.runtime = 0;
                result.timeout = false;
                return result;
            }
        }

        // Copy centroids
        centroids = new_centroids;
        
        // reset centroids
        reinit(new_centroids);

        update_centroids(dataset, new_centroids, assigned_clusters, 
        cluster_info, numCols);

        }

    // auto end = std::chrono::high_resolution_clock::now();
    // auto Totaltime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    result.loop_counter = loop_counter;
    result.num_dist = dataset.size() * loop_counter * num_clusters;
    result.assigned_labels = assigned_clusters;
    result.centroids = new_centroids;
    // result.runtime = float(Totaltime.count());
    result.timeout = false;

    return result;

}
