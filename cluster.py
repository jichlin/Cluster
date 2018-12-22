import tensorflow as tf
import numpy as np
import csv
import sklearn.decomposition as skd
from matplotlib import pyplot as plt

class SOM:
    def __init__(self,height,width,inp_dimension):
        #step 1 Initialisasi height width dimension
        self.height = height
        self.width = width
        self.inp_dimension = inp_dimension

        #step 2   Initialisasi node , placeholder input , weight

        self.node = [tf.to_float([i,j]) for i in range(height) for j in range(width)]
        self.input = tf.placeholder(tf.float32,[inp_dimension])
        self.weight = tf.Variable(tf.random_normal([height * width , inp_dimension]))

        #steo 3 fungsi cari node terdekat, update weight
        self.best_matching_unit = self.get_bmu(self.input)
        self.updated_weight = self.get_update_weight(self.best_matching_unit,self.input)

    def get_bmu(self,input):
        #hitung distance + euclidean
        expand_input = tf.expand_dims(input,0)
        euclidean = tf.square(tf.subtract(expand_input,self.weight))
        distances = tf.reduce_sum(euclidean,1)

        #cari index distance terpendek
        min_index = tf.argmin(distances,0)
        bmu_location = tf.stack([tf.mod(min_index,self.width),tf.div(min_index,self.width)])
        return tf.to_float(bmu_location)
        
    def get_update_weight(self , bmu , input):
        #inisialisasi learning rate dan sigma
        learningRate = .5
        sigma = tf.to_float(tf.maximum(self.height,self.width)/ 2)

        #hitung perbedana jarak bmu ke node lain pake euclidean
        expand_bmu = tf.expand_dims(bmu,0)
        euclidean = tf.square(tf.subtract(expand_bmu,self.node))
        distances = tf.reduce_sum(euclidean,1)

        ns = tf.exp(tf.negative(tf.div(distances**2,2*sigma**2)))

        #hitung rate
        rate = tf.multiply(ns,learningRate)
        numofNode = self.height * self.width
        rate_value = tf.stack([ tf.tile(tf.slice(rate,[i],[1]),[self.inp_dimension]) for i in range(numofNode)])
        
        #hitung update weight
        weight_diff = tf.multiply(rate_value, tf.subtract(tf.stack([input for i in range (numofNode)]),self.weight))

        updated_weight = tf.add(self.weight,weight_diff)
        return tf.assign(self.weight,updated_weight)

    def train(self, dataset , numOfEpoch):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(numOfEpoch+1):
                for data in dataset:
                    feed = {self.input : data}
                    sess.run([self.updated_weight],feed)
                self.weight_value = sess.run(self.weight)
                self.node_value = sess.run(self.node)
                cluster = [[] for i in range(self.width)]
                for i,location in enumerate(self.node_value):
                    cluster[int(location[0])].append(self.weight_value[i])
                self.cluster = cluster

def numerizeLabel(row):
    if(row[1] == "Australia"):
        label = 1
    elif(row[1] == "Belgium"):
        label = 2
    elif(row[1] == "France"):
        label = 3
    elif(row[1] == "Switzerland"):
        label = 4
    elif(row[1] == "United States"):
        label = 5
    return label

def featureSelection(row):
    data = []
    cd = float(row[3]) + float(row[4]) / 100
    dividend = float(row[7]) + float(row[8])
    divisor = float(row[6]) + float(row[10])

    if divisor == 0 :
        divisor = 1

    fr = dividend / divisor

    sg = float(row[12])
    p = float(row[14])
    salt = float(row[15])
    data.append(cd)
    data.append(fr)
    data.append(sg)
    data.append(p)
    data.append(salt)
    return data

def normalize(dataset):
    min_data = min(dataset)
    max_data = max(dataset)
    normalized = [(i - min_data) / (max_data - min_data) for i in dataset]
    return normalized

def denormalize(dataset):
    min_data = min(dataset)
    max_data = max(dataset)
    denormalized = [int(((i * (max_data - min_data )) + max_data)) for i in data]
    return denormalized   

def loadData(path):
    dataset = []
    reader = csv.reader(open(path))
    next(reader)
    for row in reader:
        dataset.append(featureSelection(row))

    cd = []
    fr = []
    sg = []
    p = []
    salt = []
    for row in dataset:
        cd.append(row[0])
        fr.append(row[1])
        sg.append(row[2])
        p.append(row[3])
        salt.append(row[4])
    
    cd = normalize(cd)
    fr = normalize(fr)
    sg = normalize(sg)
    p = normalize(p)
    salt = normalize(salt)
    i = 0
    for row in dataset:
        row[0] = cd[i]
        row[1] = fr[i]
        row[2] = sg[i]
        row[3] = p[i]
        row[4] = salt[i]
        i += 1
    

    return dataset


dataset = loadData("clustering.csv")
data = np.array(dataset)
data = skd.PCA(n_components=3).fit_transform(data)

numOfEpoch = 5000
print(data)

som = SOM(5,5,3)
som.train(data,numOfEpoch)
                    
plt.imshow(som.cluster)
plt.show()

