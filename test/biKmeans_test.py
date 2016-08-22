#################################################
# kmeans: k-means cluster
# Author : zouxy
# Date   : 2013-12-25
# HomePage : http://blog.csdn.net/zouxy09
# Email  : zouxy09@qq.com
#################################################

from numpy import *
import time
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from algrithem.biKmeans import *

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # step 1: load data
print "step 1: load data..."
dataSet = []
fileIn = open(BASE_DIR + '/data/kmeans_dataSet.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split('\t')
    dataSet.append([float(lineArr[0]), float(lineArr[1])])

# # step 2: clustering...
print "step 2: clustering..."
dataSet = mat(dataSet)
k = 4
centroids, clusterAssment = biKmeans(dataSet, k)

# # step 3: show the result
print "step 3: show the result..."
showCluster(dataSet, k, centroids, clusterAssment)
