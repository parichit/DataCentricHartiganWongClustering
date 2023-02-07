#include <iostream>
#include <vector>
#include "misc_utils.hpp"
#include "algo_utils.hpp"
#include "HW_utils.hpp"
#include <chrono>

using namespace std;


inline output_data hw_kmeans(vector<vector <float> > &dataset, int num_clusters, 
float threshold, int num_iterations, int numCols, int time_limit){

    int loop_counter = 0;
    
    vector<vector<float> > centroids(num_clusters, vector<float>(numCols, 0));
    vector<vector<float> > new_centroids(num_clusters, vector<float>(numCols, 0));
    vector<vector<float> > dist_matrix(dataset.size(), vector<float>(2, 0));
    
    vector<vector<float> > cluster_info(num_clusters, vector<float>(2, 0));  
    vector<float> an1(num_clusters);  
    vector<float> an2(num_clusters);  

    vector<int> assigned_clusters1(dataset.size(), 0);
    vector<int> assigned_clusters2(dataset.size(), 0);
 
    // Create objects
    print_utils pu;

    long long int dist_counter = 0;

    output_data result;

    // Start time counter 
    // auto start = std::chrono::high_resolution_clock::now();
    
    // Initialize centroids
    init_centroids_sequentially(centroids, dataset, num_clusters);

    calculate_distances(dataset, centroids, dist_matrix, 
    num_clusters, assigned_clusters1, assigned_clusters2, cluster_info, 
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
        an1[i] = cluster_info[i][0]/(cluster_info[i][0]-1);
        an2[i] = cluster_info[i][0]/(cluster_info[i][0]+1);
    }

    while (loop_counter < num_iterations){

        loop_counter++;

        // Calculate new centroids
        update_centroids(dataset, new_centroids, assigned_clusters1, 
        cluster_info, numCols);

        // Check Convergence
        if (check_convergence(new_centroids, centroids, threshold)){
                cout << "Convergence at iteration: " << loop_counter << "\n";
                break;
        }
        
        // OPTRANS STAGE 
        /* Find the 
        two closest centers and update, 
        assigned_clusters1, assigned_clusters2, and
        distances
        */
        optrans(dataset, new_centroids, dist_matrix, assigned_clusters1, assigned_clusters2, 
        cluster_info, an1, an2, num_clusters, dist_counter);

        // reset centroids
        reinit(new_centroids);

        // Update centroids
        update_centroids(dataset, new_centroids, assigned_clusters1, 
        cluster_info, numCols);

        if (check_status(centroids, new_centroids) == false){
            break;
        }

        // QUICKTRANS STAGE 
        /* Transffer the point between two closest 
        centers, if SSE changes
        */
        quicktrans(dataset, new_centroids, dist_matrix, assigned_clusters1, assigned_clusters2, 
        cluster_info, an1, an2, num_clusters, dist_counter);

        if (num_clusters == 2){
            cout << "Algorithm was started with two clusters, not entering OPTRANS" << endl;
            break;
        }

        // Copy centroids
        centroids = new_centroids;

        // reset centroids
        reinit(new_centroids);

        // Update centroids
        // update_centroids(dataset, new_centroids, assigned_clusters1, 
        // cluster_info, numCols);

    }

    // auto end = std::chrono::high_resolution_clock::now();
    // auto Totaltime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    result.loop_counter = loop_counter;
    result.num_dist = dataset.size() * loop_counter * num_clusters;
    result.assigned_labels = assigned_clusters1;
    result.centroids = new_centroids;
    // result.runtime = float(Totaltime.count());
    result.timeout = false;

    return result;

}
