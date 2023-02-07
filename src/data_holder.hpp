#include <vector>

using namespace std;

struct output_data {
    int loop_counter = 0;
    int num_dist = 0;
    vector<int> assigned_labels;
    vector<vector <float>> centroids;
    float runtime = 0;
    bool timeout = false;
};