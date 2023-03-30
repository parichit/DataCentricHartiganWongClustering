#include <vector>

using namespace std;

struct output_data {
    int loop_counter = 0;
    unsigned long long int num_dist = 0;
    vector<int> assigned_labels;
    vector<vector <float>> centroids;
    float runtime = 0;
    bool timeout = false;
    float sse = std::numeric_limits<float>::max();
};

struct best_results{
    long long int num_dist = 0;
    vector<vector<float> > centroids;
    float sse = 0;
};