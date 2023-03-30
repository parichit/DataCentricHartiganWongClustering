#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include "data_holder.hpp"
#include "algo_utils.hpp"
#include "IOutils.hpp"
#include "dchw.hpp"
#include <chrono>

using namespace std;

// Parameters are as follows
// dataset: is the data (2D matrix format)
// num_clusters, num_iterations and threshold are self-explanatory
// note that if you want the algorithsm to stop only when
// centroids are same then pass 0 as the value for threshold.

// time_limit helps to stop the program execution if it's taking
// too long to execute. Pass a value in ms, for example, to allow the
// program to run only for 1 minute, pass 60000 as the value of time_limit.
// if the program timeouts, then it will return a large value for SSE so it can be
// ignored from the calculations

// Added num_restarts and bint as mentioned as the last two parameters.

best_results dchw_rr(vector<vector <float> > &dataset, int num_clusters,
float threshold, int num_iterations, int numCols, int time_limit,
int num_restarts, int bint, string init_type, vector<int> indices){

    float best_score = std::numeric_limits<float>::max();

    // We will update the following structure
    // everytime we get a better clustering i.e., SSE etc.
    output_data results;
    best_results best;

    // cout << "clusters: " << num_clusters << endl;
    // cout << "restarts:" << num_restarts << endl;
    // cout << "threshold:" << threshold << endl;
    // cout << "iter:" << num_iterations << endl;
    // cout << "cols:" << numCols << endl;
    // cout << "lim:" << time_limit << endl;
    // cout << "seed:" << bint << endl;

    // vector<int> temp = {5, 10, 20, 30, 40};

    // Run the program iteratively
    for (int i=0; i < num_restarts; i++){

        vector<int> indices = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14};

        // for(int j=0; j < indices.size(); j++){
        //     indices[j] = indices[j] + i;
        // }

        results = dchw_kmeans(dataset, num_clusters, threshold, num_iterations, 
        numCols, time_limit, init_type, indices, bint+i);

        // cout << "run :" << i << " SSE: " << results.sse << endl;
        if (results.sse < best_score){
            best_score = results.sse;
            best.num_dist = results.num_dist;
            best.centroids = results.centroids;
            best.sse = best_score;
            //cout << best.centroids << endl;
        }
    }

    // Now this structure called best contains 4 things and it can be
    // accessed by keys
    // 1. To access best score (SSE), best.best_score
    // 3. best calculations -> best.best_calcs
    // 4. best assignments -> best.centroids
    // You can see the structure on line 15-21 in this file.
    return best;
}

// Order of command line arguments
// 1. File Path
// 2. Number of clusters
// 3. Number of iterations
// 4. Number of restarts
// 5. bint
// 6. Output Dir Path (where the output files be created)

// Ensure that file path is always the first parameter
int main(int argc, char* argv[]){

// Read file path
string filePath = argv[1];
string index = filePath.substr(filePath.find_last_of('_') + 1);

// cout << filePath << endl;

// clusters (convert string to integer)
int num_clusters = atoi(argv[2]);

// Hard coding based on RR file that was shared
float threshold = 0.001;

int num_iterations = atoi(argv[3]);
int num_restarts = atoi(argv[4]);
int bint = atoi(argv[5]);

// A path where you want to write the files
string outPath = argv[6];

// Change the time limit below (if need be), currently it is 2 hours (120 minutes)
// If a single run of DCHW or HW does not complete within the 2 hours then it returns with 
// a very large value of SSE and will be automatically ignored in your pipeline.
best_results best;
int time_limit = 7200000;

// Load the data into dataset
vector<vector <float> > dataset;
vector<int> labels;


// I used the file_1.csv as sample for this wrapper and it's good that it does not have labels in the file.
// CONSTRAINT: The labels (if present) MUST be present in the last column of the file.
// The last two boolean flags are TRUE when the file contains both column-names and labels
// If these are TRUE then the file is read by avoiding the first row (header/column names)
// and last column i.e. labels

// Example, if the file only contains last column which is the label but there is no HEADER then
// pass FALSE, TRUE to indicate -- don't remove first row and remove the last column.
// The following line assume that niether HEADER nor labels are present in the file. 

std::pair<int, int> p = readSimulatedData(filePath, dataset, labels, false, false);
int numCols = p.second;


// Start timer
auto t1 = std::chrono::high_resolution_clock::now();

// cout << "clusters: " << num_clusters << endl;
// cout << "restarts:" << num_restarts << endl;
// cout << "threshold:" << threshold << endl;
// cout << "iter:" << num_iterations << endl;
// cout << "cols:" << numCols << endl;
// cout << "lim:" << time_limit << endl;
// cout << "seed:" << bint << endl;


vector<int> my_indices = {2, 5, 7, 8, 10};

// Call the function
best = dchw_rr(dataset, num_clusters, threshold, num_iterations, numCols,
time_limit, num_restarts, bint, "indices", my_indices);


// Calculate elapsed time in seconds
auto t2 = std::chrono::high_resolution_clock::now();
auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
float runtime = elapsed_time.count()/1000.0;;

// Save to files (phewwwww - this is not that smooth to do in CPP)
// Write the best assignments (each assignment is seperated by a comma)

// for (int i = 0; i<best.centroids.rows(); i++){
//     for(int j = 0; j< best.centroids.cols(); j++){
//         cout << best.centroids(i, j) << "," ;
//     }
//     // outfile << "\n";
//     cout << "\n";
// }

ofstream outfile;
outfile.open(outPath+"file2_" + index, ios::trunc);

for (int i = 0; i < best.centroids.size(); i++){
    for(int j = 0; j < best.centroids[i].size(); j++){
        outfile << best.centroids[i][j] << ",";
        if (j != best.centroids[i].size()-1){
            outfile << ",";
        }
    }
    outfile << endl;
}
outfile.close();

// Write the best calculations
// Since calculation is a single number so writing it
// in the text file because there is nothing to separate by commas
outfile.open(outPath+"file3_" + index, ios::trunc);
outfile << best.num_dist << endl;
outfile.close();

// Writing the time
outfile.open(outPath+"file4_" + index, ios::trunc);
outfile << runtime << endl;
outfile.close();

// I think that's it. I think there could be minor issues.
// I can fix major issues if they interfere with running the pipeline.

}