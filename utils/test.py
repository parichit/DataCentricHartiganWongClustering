import pandas as pd
from sortedcontainers import SortedList, SortedDict
import numpy as np
from sympy import Point, Line, Plane, Segment
from utils.dataIO import *
import seaborn as sns
import matplotlib.pyplot as plt

a = [[45, 6], [34, 7], [98, 0], [1, 2], [12, 67], [0, 74]]
# b = SortedDict(a)
#
# print(b, b.peekitem(len(b)-1)[0])

# for i in b:
#     print(i[0])

# m = np.array(list(b.islice(3, )))
# print(m)
# print(m[:,1])

# b = SortedDict(a)
# print(b)
# print(b.peekitem(len(b)-1)[0])


a = np.array([[1,1], [2,1], [1, 3], [0, 1]])
print(a, a[[1, 0], ])
#
# o = np.array([1,2,0,1,0])
# p = np.array([1,2,1,1,0])
# m = np.where(o < 1)[0]
# print(m)
# m = m[m!=2]
# print(m)
#
# a = np.array([1,2,3,4,5,6,7,8,9,10])
# a[[5, 2, 1]] = [0, 0, 0]
# print(a[[5, 2, 1]])

# a = pd.DataFrame({'pc1': [1,2,3,4,5], 'pc2':[6,7,8,9,7], 'labels':['a', 'a', 'a', 'a', 'b']})
# # print(a.loc[a['labels']=='b'][['pc1', 'pc2']].values[0])
#
# a.iloc[0, a.shape[1]-1] = 'm'
# print(a)
#
# a = np.array([1.001, 2, 3])
# b = np.array([1, 2.001, 3])
#
# print(np.cross(a, b))

a = {}

# for i in range(3):
#     for j in range(3):
#
#         if i != j:
#
#             if i not in a.keys():
#                 a[i] = {j:123}
#             else:
#                 a[i][j] = 123
#
# print(a)

file_list = ['test_100_3_3.csv']
file_path = os.path.join(Path(__file__).parents[1], "sample_data")

for data_file in file_list:
    data, labels = read_simulated_data(os.path.join(file_path, data_file))
    data = pd.DataFrame(data)
    centroids = np.array([[10.4621, 13.131,  13.6524],
                 [11.2961, 12.4165, 13.6772],
                 [11.4689, 11.0801, 12.143]])
    centroids = pd.DataFrame(centroids, columns=[[0, 1, 2]])

    data = data.append(centroids, ignore_index=True)

    labels += [3, 3, 3]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x = data[0]
    # y = data[1]
    # z = data[2]

    # ax.scatter(x, y, z, c=labels)
    # plt.show()

def find_sur(x, y):
    # return (32 + (8*x) - (2*y))/17
    return  (6 + 2 * x)/4
    # return (16 * x + 16 * y)/32

def find_sur_n(x, y):
    return (2*x + 8*y - 26)


# a = Point(1, 3, 2)
# b = Point(5, 3, 4)
# m = a.midpoint(b)
# l = Line(a, b)
# l2 = l.perpendicular_line(m)

# x = np.linspace(1, 5, 50)
# y = np.linspace(1, 5, 50)
# X, Y = np.meshgrid(x, y)
# Z = find_sur(X, Y)
#
# pl = Plane(a, b, l2.points[0])
# print(pl.equation(), m, l2)

# perp_plane = pl.perpendicular_plane(l2.points[0], l2.points[1])
# nv = perp_plane.normal_vector
#
# print(perp_plane.equation())
#
# print(np.dot(nv, a-list(m)), np.dot(nv, b-list(m)))
#
# Z1 = (18 - 4*X)/2
#
# t = np.array([[1, 3, 2], [5, 3, 4], list(m), list(l2.points[0])])
# test = pd.DataFrame(t, columns=[[0, 1, 2]])
# labels = [0, 1, 2, 3]

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# p1 = test[0]
# p2 = test[1]
# p3 = test[2]
# ax.scatter(p1, p2, p3, c=labels, s=100)
# ax.plot_surface(X, Y, Z)
# ax.plot_surface(X, Y, Z1)
# plt.show()



# a = np.array([1,2,3])
# b = np.array([0,2,3])
# print(np.where(a != b)[0])
# print(list(range(4, -1, -1)))
# if a.size > 0:
#     print("hello")

# print(b)