#include <iostream>
#include <vector>
#include "misc_utils.hpp"
#include "algo_utils.hpp"
#include "HW_utils.hpp"
#include "dchw_utils.hpp"
#include <chrono>

using namespace std;


inline output_data dchw_kmeans(vector<vector <float> > &dataset, int num_clusters, 
float threshold, int num_iterations, int numCols, int time_limit){

    vector<vector<float> > centroids(num_clusters, vector<float>(numCols, 0));
    vector<vector<float> > new_centroids(num_clusters, vector<float>(numCols, 0));
    vector<float> dist_matrix(dataset.size(), 0);
    
    vector<vector<float> > cluster_info(num_clusters, vector<float>(2, 0)); 
    vector<float> cluster_radius(num_clusters, 0.0);
    vector<float> an1(num_clusters);  
    vector<float> an2(num_clusters);  

    vector<int> assigned_clusters(dataset.size(), 0);
    vector<vector<float> > he_data;

    long long int dist_counter = 0;
    int loop_counter = 0, i = 0, curr_clsuter = 0;

    output_data result;
    bool centroid_status = false;

    // Start time counter 
    // auto start = std::chrono::high_resolution_clock::now();
    
    // Initialize centroids
    init_centroids_sequentially(centroids, dataset, num_clusters);

    calculate_distances(dataset, centroids, dist_matrix, 
    num_clusters, assigned_clusters, cluster_info, 
    dist_counter);

    // Check for empty clusters and return
    for (i=0; i<num_clusters; i++){
    
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
    for (i = 0; i < num_clusters; i++){
        if (cluster_info[i][0] > 1){
            an1[i] = cluster_info[i][0]/(cluster_info[i][0]-1);
            cluster_info[i][1] *= an1[i];
        }
        
        an2[i] = cluster_info[i][0]/(cluster_info[i][0]+1);
        // cluster_info[i][1] *= an1[i];
    }

    // Calculate new centroids
    update_centroids(dataset, new_centroids, assigned_clusters, 
    cluster_info, numCols);

    // print_2d_vector(centroids, centroids.size(), "Initial");

    print_2d_vector(cluster_info, cluster_info.size(), "cluster_info");


    while (loop_counter < num_iterations){

        loop_counter++;

        cout << loop_counter << endl;
        // print_2d_vector(new_centroids, new_centroids.size(), "Inside loop");
        print_2d_vector(cluster_info, cluster_info.size(), "cluster_info");

        for (int clus = 0 ; clus < num_clusters ; clus++){

                // Check Convergence
                if (check_status(centroids, new_centroids)){

                    centroid_status =  true;

                    // Find all the HE points for the current cluster

                    get_he_data(dataset, dist_matrix, new_centroids[clus], 
                    an1, clus, assigned_clusters, num_clusters, cluster_info, he_data, dist_counter);

                    // cout << "loop: " << loop_counter << "\t" << he_data.size() << endl;

                    // Find the closest centroids to the current HE points and assign them
                    // accordingly
                    if(he_data.size() > 0){
                        reassign_he_data(dataset, new_centroids, cluster_info, he_data, 
                        assigned_clusters, clus, dist_matrix, an1, an2, dist_counter);
                    }

                }
                
                else{
                centroid_status =  false; 
                }
            }

            if (centroid_status == false){
                cout << "Convergence at iteration: " <<  loop_counter << endl;
                break;
            }

            // Copy centroids
            centroids = new_centroids;
            
            // reset centroids
            reinit(new_centroids);

            update_centroids(dataset, new_centroids, assigned_clusters, 
            cluster_info, numCols);

            // Update cluster info
            update_cluster_info(assigned_clusters, cluster_info, dist_matrix, an1, an2, 
            num_clusters);

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
