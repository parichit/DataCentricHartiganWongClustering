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
    vector<vector <float> > center_dist_mat(num_clusters, vector<float>(num_clusters, 0));
    vector<float> dist_mat(dataset.size(), 0);
    vector<vector<int> > neighbors(num_clusters);

    vector<vector<vector <float> > > cutoff_points(num_clusters, vector<vector<float> >(num_clusters, vector<float>(numCols, 0)));
    vector<vector<vector <float> > > affine_vectors(num_clusters, vector<vector<float> >(num_clusters, vector<float>(numCols, 0)));
    
    vector<vector<float> > cluster_info(num_clusters, vector<float>(3, 0)); 
    vector<float> cluster_safe(num_clusters, 0.0);
    vector<float> an1(num_clusters);  
    vector<float> an2(num_clusters);  

    vector<int> assigned_clusters(dataset.size(), 0);
    vector<vector<float> > he_data;

    int loop_counter = 0, i = 0, j = 0, my_cluster = 0, new_clus = 0;
    float ot_dist = 0, ot_dist_w = 0, my_dist = 0, my_dist_w = 0, shor_dist = 0, temp = 0;
    long long int dist_counter = 0;

    output_data result;
    bool centroid_status = false;

    float neighbor_time = 0, inner_loop_time = 0;
    float lim = std::numeric_limits<float>::max();

    // Start time counter 
    // auto start = std::chrono::high_resolution_clock::now();
    
    // Initialize centroids
    init_centroids_sequentially(centroids, dataset, num_clusters);

    calculate_distances(dataset, centroids, dist_mat, 
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
        }
        an2[i] = cluster_info[i][0]/(cluster_info[i][0]+1);
    }

    // Calculate new centroids
    update_centroids(dataset, new_centroids, assigned_clusters, 
    cluster_info, numCols);
    
    // print_2d_vector(new_centroids, new_centroids.size(), "Initial");
    // print_2d_vector(cluster_info, cluster_info.size(), "cluster_info");

    float shortest_sse = 0;

    while (loop_counter < num_iterations){

        // cout << loop_counter << endl;
        // print_2d_vector(new_centroids, new_centroids.size(), "Inside loop");
        // print_2d_vector(cluster_info, cluster_info.size(), "cluster_info");

         // Check Convergence
        if (check_status(centroids, new_centroids)){
            
            loop_counter++;
            // cout << loop_counter << endl;

            // Set current centroid status
            centroid_status = true;
            
            
            // Find neighbors
            // auto t1 = std::chrono::high_resolution_clock::now();
            find_neighbors(new_centroids, center_dist_mat, cluster_info, neighbors, 
            cutoff_points, affine_vectors, an1, an2, cluster_safe);
            // auto t2 = std::chrono::high_resolution_clock::now();
            // auto temp = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            // neighbor_time += temp;

            // print_2d_vector(new_centroids, new_centroids.size(), "Initial");
            // print_2d_vector(cluster_info, cluster_info.size(), "cluster_info");
            // print_2d_vector(neighbors, neighbors.size(), "neighbors");
            // print_vector(cluster_safe, cluster_safe.size(), "cluster_safe");
            // print_2d_vector(cutoff_points[0], cutoff_points[0].size(), "cutoff_points");

            // auto t3 = std::chrono::high_resolution_clock::now();
            for (i = 0; i < dataset.size(); i++){

                my_cluster = assigned_clusters[i];

                // my_dist = calc_sq_dist(dataset[i], new_centroids[my_cluster]);
                // if (sqrt(my_dist) < cluster_safe[my_cluster]){
                //     continue;
                // }

                // Loop through the neighbors (if they exist)
                if ((cluster_info[my_cluster][0] > 1) && (cluster_info[my_cluster][2] > 0)){
                        // continue;

                    my_dist = calc_sq_dist(dataset[i], new_centroids[my_cluster]);
                    my_dist_w = my_dist * an1[my_cluster]; 
                    shortest_sse = lim;
                    temp = sqrt(my_dist);

                    if(my_dist > cluster_info[my_cluster][1])
                        cluster_info[my_cluster][1] = my_dist;

                    // auto t3 = std::chrono::high_resolution_clock::now();
                    for (j=0; j<neighbors[my_cluster].size(); j++){ 

                        // Perform computations only if HE point
                        // if (find_context_direction(dataset[i], affine_vectors[my_cluster][neighbors[my_cluster][j]], 
                        // cutoff_points[my_cluster][neighbors[my_cluster][j]], ot_dist)){
                        if ( ( an2[neighbors[my_cluster][j]] * abs(temp - center_dist_mat[my_cluster][neighbors[my_cluster][j]])) 
                        > (an1[my_cluster] * temp) ) {
                            continue;
                        }

                        ot_dist = calc_sq_dist(dataset[i], new_centroids[neighbors[my_cluster][j]]);
                        // shor_dist = sqrt(ot_dist);
                        ot_dist_w = ot_dist * an2[neighbors[my_cluster][j]];

                        if(ot_dist > cluster_info[neighbors[my_cluster][j]][1])
                            cluster_info[neighbors[my_cluster][j]][1] = ot_dist;


                        if(ot_dist_w < shortest_sse){
                            shortest_sse = ot_dist_w;
                            new_clus = neighbors[my_cluster][j];
                            // cout << "Point: " << i << " Old: " << my_cluster << " New: " << new_clus << endl;
                        } 

                        //}

                    }

            }
                    
                    // auto t4 = std::chrono::high_resolution_clock::now();
                    // auto temp1 = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
                    // inner_loop_time += temp1;

                    if (shortest_sse < my_dist_w){
                            cluster_info[my_cluster][0] = cluster_info[my_cluster][0] - 1;
                            assigned_clusters[i] = new_clus;
                            cluster_info[new_clus][0] = cluster_info[new_clus][0] + 1;
                            // cout << "HE point: " << i << " Old clus: " << my_cluster << " New clus: " << new_clus << endl;
                    }

                }
                // auto t4 = std::chrono::high_resolution_clock::now();
                // auto temp1 = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
                // inner_loop_time += temp1;

            for (i = 0; i < num_clusters; i++){
                cluster_info[i][1] = sqrt(cluster_info[i][1]);
                if (cluster_info[i][0] > 1)
                    an1[i] = cluster_info[i][0]/(cluster_info[i][0]-1);
                an2[i] = cluster_info[i][0]/(cluster_info[i][0]+1);
            }

        }
                
        else{
            centroid_status =  false; 
            }

        if (centroid_status == false){
                cout << "DCHW convergence at iteration: " <<  loop_counter << endl;
                break;
        }

        // Copy centroids
        centroids = new_centroids;
        
        // reset centroids
        reinit(new_centroids);

        update_centroids(dataset, new_centroids, assigned_clusters, 
        cluster_info, numCols);

        // Update cluster info
        // update_cluster_info(assigned_clusters, cluster_info, dist_mat, an1, an2, 
        // num_clusters);

        }

    // auto end = std::chrono::high_resolution_clock::now();
    // auto Totaltime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    result.loop_counter = loop_counter;
    result.num_dist = dataset.size() * loop_counter * num_clusters;
    result.assigned_labels = assigned_clusters;
    result.centroids = new_centroids;
    // result.runtime = float(Totaltime.count());
    result.timeout = false;

    // cout << "neighbor time: " << neighbor_time << endl;
    // cout << "Inner Loop time: " << inner_loop_time << endl;

    return result;

}
