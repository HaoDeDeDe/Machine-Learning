import numpy as np
import matplotlib.pyplot as plt
import random


def Distortion(K, centroids, assign, data):
    distort = 0
    for k in range(K):
        distort += np.sum((data[assign[k],]-centroids[k,]) * (data[assign[k],]-centroids[k,]))
    return(distort)

def Centroids_Random_Init(K, data):
    n, d = np.shape(data)
    data_range = [np.min(data), np.max(data)]
    centroids = np.random.rand(K, d) * (data_range[1] - data_range[0]) + data_range[0]
    return(centroids)

def Centroids_PP_Init(K, data):
    n, d = np.shape(data)
    centroids = np.zeros((K,d))
    for k in range(K):
        if k==0:
            centroids[k,] = data[np.random.choice(range(n),size=1,replace=False),]
        else:
            dist = np.zeros((n,k))
            for i in range(k):
                dist[:,i] = np.sqrt(np.sum((data-centroids[i,])*(data-centroids[i,]),axis=-1))
            D = np.min(dist, axis=-1)
            prob = D*D / sum(D*D)
            centroids[k,] = data[np.random.choice(range(n),size=1,replace=False,p=prob),]
    return(centroids)

def Cluster(K, centroids, data):
    n, d = np.shape(data)
    assign = []
    distort = []
    for k in range(K):
        assign.append([])
    nochange = 0
    while (nochange != K):
        # wipe previous assignment
        for i in range(K):
            del assign[i][:]
        # make new assignment
        for i in range(n):
            k = np.argmin(np.sum((data[i,] - centroids) * (data[i,] - centroids), axis=-1))
            assign[k].append(i)
        # update centroids
        nochange = 0
        for k in range(K):
            if len(assign[k]) != 0:
                new_cent = np.sum(data[assign[k],], axis=0) / len(assign[k])
                if np.array_equal(new_cent, centroids[k,]):
                    nochange += 1
                centroids[k,] = new_cent
            else:
                nochange += 1
        distort.append(Distortion(K, centroids, assign, data))
    return ([centroids, assign, distort])


def Kmeans(K, data):
    # initialize centroids
    centroids = Centroids_Random_Init(K, data)
    # cluster
    return(Cluster(K, centroids, data))


def KmeansPP(K, data):
    # initialize centroids
    centroids = Centroids_PP_Init(K, data)
    # cluster
    return (Cluster(K, centroids, data))



if __name__ == '__main__':
    data = np.loadtxt("/Users/DaisyYang/Desktop/STAT_37710/Assignment/1/Kmeans/toydata.txt")
    random.seed(100)

    centroids, assign, distort = Kmeans(3, data)
    plt.scatter(data[assign[0],0], data[assign[0],1], color="red")
    plt.scatter(data[assign[1],0], data[assign[1], 1], color="blue")
    plt.scatter(data[assign[2], 0], data[assign[2], 1], color="green")
    plt.title("Kmeans: Cluster Assignment of Points")
    plt.show()

    for i in range(20):
        centroids, assign, distort = Kmeans(3, data)
        plt.plot(range(len(distort)), distort)
        print(distort[-1])
    plt.xlabel("iteration")
    plt.ylabel("distortion")
    plt.title("Kmeans: Value of Distortion as a Function of Iteration in 20 Runs")
    plt.show()

    for i in range(20):
        centroids, assign, distort = KmeansPP(3, data)

        plt.plot(range(len(distort)), distort)
        print(distort[-1])
    plt.xlabel("iteration")
    plt.ylabel("distortion")
    plt.title("Kmeans++: Value of Distortion as a Function of Iteration in 20 Runs")
    plt.show()



