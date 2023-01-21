## Basic Example

- A simple example can be seen in run_test/test_he_data.py.

- On line 62, the normal HW algorithm is called and SSE is the last parameter returned.

- on line 67 is the data-centric version being called and SSE is the last parameter.

#### Note

- This implementation produces exaclty the same answers between the normal and data centric version for k = 2 - 12. On some datasets, for large values of K, there are differences in the final centroids obtained from the two algorithms.