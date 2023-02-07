#include <iostream>
#include <string>
#include "data_holder.hpp"
#include "algo_utils.hpp"
#include "IOutils.hpp"
#include "hw.hpp"
#include "dchw.hpp"
#include <chrono>

#include <filesystem>

using namespace std;


//  string basePath = "/nobackup/parishar/DATASETS/";
string basePath = "/Users/schmuck/Library/CloudStorage/OneDrive-IndianaUniversity/Box Sync/PhD/DATASETS/";
// string basePath = "/Users/schmuck/Documents/OneDrive - Indiana University/Box Sync/PhD/Data-Centric-KMeans/";

int main(){

    // Experiment specific path
    // string filePath = basePath + "clustering_data/";
    // string filePath = basePath + "dims_data/";
    // string filePath = basePath + "scal_data/";
    // string filePath = basePath + "sample_data/";
     basePath = basePath + "real_data/experiment_data/";

    
    string centroidFilePath = "";
    string somefilePath = "";

    std::vector<vector <float> > dataset;
    vector<int> labels;

    
    // Declare variables
    int num_iterations = 100;
    float threshold = 0;

    // vector<string> file_list = {"fourclass.csv", "codrna.csv", "Breastcancer.csv",
    // "crop.csv", "Census.csv"};

    // vector<string> out_list = {"fourclassCentroids", "codrnaCentroids", "BreastcancerCentroids",
    // "cropCentroids", "CensusCentroids"};

    vector<string> file_list = {"50_2_10.csv"};
    vector<string> out_list = {"50_2_10"};
    int num_clusters = 5;

    somefilePath = basePath + file_list[0];

    cout << somefilePath << endl;
    
    // Read in the data
    auto t1 = std::chrono::high_resolution_clock::now();
    std::pair<int, int> p = readSimulatedData(somefilePath, dataset, labels, true, true);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto file_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    
    cout << "File reading: " << file_int.count() << " MS\n";
    cout << "Data size: " << dataset.size() << " X " << dataset[0].size() << endl;
    
    int numRows = p.first;
    int numCols = p.second;

    output_data res;
    
    // Read the centroids
    // vector<vector<float> > centroids(num_clusters, vector<float>(numCols, 0.0));
    
    // centroidFilePath = basePath + out_list[0] + "_" + std::to_string(num_clusters) + "_0" + ".txt" ;
    cout << centroidFilePath << endl;
    
    // read_kplus_plus_centroids(centroidFilePath, centroids, num_clusters);

    // Debug - Testing
    // cout << "\nAlgo: HWKMeans," << " Clusters: " << num_clusters << ", Threshold: " << threshold << endl;
    // auto t3 = std::chrono::high_resolution_clock::now();
    // res = hw_kmeans(dataset, num_clusters, threshold, num_iterations, numCols, 60000);
    // auto t4 = std::chrono::high_resolution_clock::now();
    // auto km_int = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);
    // std::cout << "Total HWKMeans time: " << km_int.count() << " milliseconds\n";

    // print_2d_vector(res.centroids, res.centroids.size(), "Out");

    // // vector<vector<float> > rcentroids(num_clusters, vector<float>(numCols, 0.0));
    // // read_kplus_plus_centroids(centroidFilePath, rcentroids, num_clusters);

    cout << "\nAlgo: DCHW_Kmeans," << " Clusters: " << num_clusters << ", Threshold: " << threshold << endl;
    auto t5 = std::chrono::high_resolution_clock::now();
    res = dchw_kmeans(dataset, num_clusters, threshold, num_iterations, numCols, 60000);
    auto t6 = std::chrono::high_resolution_clock::now();
    auto ms_int2 = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5);
    std::cout << "Total DCHW_Kmeans time: " << ms_int2.count() << " milliseconds\n";

    
    // for(int i=2; i<=10; i++){

    //     num_clusters = i;
    //     // vector<vector<float> > km_centers(num_clusters, vector<float>(numCols, 0.0));
    //     // vector<int> km_assign(dataset.size());
    //     output_data res;

    //     cout << "\nAlgo: DCKM," << " Clusters: " << num_clusters << ", Threshold: " << threshold << endl;
    //     auto t5 = std::chrono::high_resolution_clock::now();
    //     res = hw_kmeans(dataset, num_clusters, threshold, num_iterations, numCols, 60000);
    //     auto t6 = std::chrono::high_resolution_clock::now();
    //     auto ms_int2 = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5);
    //     std::cout << "Total DCKmeans time: " << ms_int2.count() << " milliseconds\n";

    //     cout << "\nAlgo: HWKMeans," << " Clusters: " << num_clusters << ", Threshold: " << threshold << endl;
    //     auto t3 = std::chrono::high_resolution_clock::now();
    //     res = hw_kmeans(dataset, num_clusters, threshold, num_iterations, numCols, 60000);
    //     auto t4 = std::chrono::high_resolution_clock::now();
    //     auto km_int = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);
    //     std::cout << "Total HWKMeans time: " << km_int.count() << " milliseconds\n";

    // }

return 0;
}
