#Import required python modules
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random
from scipy.stats import multivariate_normal

def plotPCAImage(convertedInput):
  plt.title('Test Data') 
  plt.xlabel('Input Dimension 1') 
  plt.ylabel('Input Dimension 2') 
  plt.xticks() 
  plt.yticks() 
  plt.scatter(convertedInput[:,0],convertedInput[:,1],color='blue',facecolors='None')
  #plt.show()
  plt.savefig("PCA.png")

def PCATest():
  # load data from the text
  #data = np.loadtxt('/content/drive/My Drive/Machine Learning Algorithm/PCA/data.txt')
  data = np.loadtxt('D:/Academic life/4-2/Machine Learning Sessional/Offline 3/data.txt')
  data = data[:] 
  #data_std = data
  # standardized data
  dataStandard = StandardScaler().fit_transform(data)
  
  # co-variance matrix
  meanData = np.mean(dataStandard, axis=0)
  covarianceData = (dataStandard - meanData).T.dot((dataStandard - meanData)) / (dataStandard.shape[0]) 
  
  # eigen value and eigen vector
  eigenValueData, eigVectorData = np.linalg.eig(covarianceData)
  #print('Eigenvectors \n%s' %eig_vecs)
  #print('\nEigenvalues \n%s' %eig_vals)

  # Make a list of (eigenvalue, eigenvector) tuples
  eigPairData = [(np.abs(eigenValueData[i]), eigVectorData[:,i]) for i in range(len(eigenValueData))]

  # Sort the (eigenvalue, eigenvector) tuples from high to low
  eigPairData.sort(key=lambda x: x[0], reverse=True)

  # stack the variables horizontally
  matrixHorizontally = np.hstack((eigPairData[0][1].reshape(-1,1), eigPairData[1][1].reshape(-1,1)))

  # plot the image
  convertedInput = np.dot(dataStandard,matrixHorizontally)
  #print(convertedInput)
  plotPCAImage(convertedInput)

  # axis
  dimensionX = convertedInput[:,0]
  dimensionY = convertedInput[:,1]

  return dimensionX,dimensionY,convertedInput[:,0:2]

def calculateLogLikelihood(data,numOfCluster,dimension,meanData,covarianceData,weightData):
  currLogLiklihood = 0.0 
  
  for val in data:
    tempTotal = 0.0
    for cluster in range(numOfCluster):
      temp = np.array(val)-meanData[cluster]
      tempExponenetTerm = np.dot(temp.T, np.dot(np.linalg.inv(covarianceData[cluster]), temp))
      tempExponenetTerm = -(1/2)*tempExponenetTerm 
      normalizationConstant = 1/(np.power(np.power(2*np.pi,dimension) * np.linalg.det(np.array(covarianceData[cluster])),.5))
      gaussianDistribution = normalizationConstant * np.exp(tempExponenetTerm)
      tempTotal += weightData[cluster] * gaussianDistribution

    currLogLiklihood += np.log(tempTotal)

  return currLogLiklihood

def calculateEstep(data,numOfCluster,dimension,meanData,covarianceData,weightData):
  # initialize probability of generation of x
  probabilityData = np.zeros((len(data),numOfCluster))
  #print("Probability Inital shape:",probabilityData.shape)
  valIndices = 0
  for val in data:
    tempTotal = 0.0
    for cluster in range(numOfCluster):
      # numerotor
      temp = np.array(val)-meanData[cluster]
      tempExponenetTerm = np.dot(temp.T, np.dot(np.linalg.inv(covarianceData[cluster]), temp))
      tempExponenetTerm = -(1/2)*tempExponenetTerm 
      normalizationConstant = 1/(np.power(np.power(2*np.pi,dimension) * np.linalg.det(np.array(covarianceData[cluster])),.5))
      gaussianDistribution = normalizationConstant * np.exp(tempExponenetTerm)
      probabilityData[valIndices][cluster] = weightData[cluster] * gaussianDistribution
    valIndices +=1

  # denominator
  rowSumVal = probabilityData.sum(axis=1)[:, np.newaxis]
  probabilityData = probabilityData/rowSumVal

  return probabilityData

def calculateMStep(data,numOfCluster,dimension,meanData,covarianceData,weightData,probabilityData):
  # denominator
  sumOfProbability = np.sum(probabilityData,axis=0)
  
  for cluster in range(numOfCluster):
    # updated mean
    tempMean = 0.0
    for val in range(len(data)):
      tempMean += probabilityData[val][cluster] * data[val]
    meanData[cluster] = tempMean/sumOfProbability[cluster]
    

    # updated weight
    weightData[cluster] = sumOfProbability[cluster]/len(data)

    # updated co-variance
    tempCovariance = np.zeros((dimension,dimension))
    for val in range(len(data)):
      temp = np.array(data[val]-meanData[cluster])
      multiplication = np.dot(temp,temp.T)
      #tempCovariance += np.dot(probabilityData[val][cluster],multiplication)
      tempCovariance += (probabilityData[val][cluster]*np.outer(data[val]-meanData[cluster],data[val]-meanData[cluster]))
    covarianceData[cluster] = tempCovariance/sumOfProbability[cluster]
  
  
  return np.array(meanData),np.array(covarianceData),np.array(weightData)

def gaussianDistributionofCluster(data,numOfCluster,dimension,meanData,covarianceData,weightData):
  probabilityData = np.zeros((len(data),numOfCluster))
  #print("Probability Inital shape:",probabilityData.shape)
  valIndices = 0
  for val in data:
    tempTotal = 0.0
    for cluster in range(numOfCluster):
      # numerotor
      temp = np.array(val)-meanData[cluster]
      tempExponenetTerm = np.dot(temp.T, np.dot(np.linalg.inv(covarianceData[cluster]), temp))
      tempExponenetTerm = -(1/2)*tempExponenetTerm 
      normalizationConstant = 1/(np.power(np.power(2*np.pi,dimension) * np.linalg.det(np.array(covarianceData[cluster])),.5))
      gaussianDistribution = normalizationConstant * np.exp(tempExponenetTerm)
      probabilityData[valIndices][cluster] = gaussianDistribution
    valIndices +=1

  return probabilityData


def plotEMAlgo(data,numOfCluster,dimension,meanData,covarianceData,weightData):
  currentGaussianDistribution = gaussianDistributionofCluster(data,numOfCluster,dimension,meanData,covarianceData,weightData)
  #print(currentGaussianDistribution.shape)
  clusterX = [[] for cluster in range(numOfCluster)]
  clusterY = [[] for cluster in range(numOfCluster)]

  for i in range(len(data)):
    clustersInData = list(currentGaussianDistribution[i])
    maxValueIndex = clustersInData.index(max(clustersInData))
    clusterX[maxValueIndex].append(data[i][0])
    clusterY[maxValueIndex].append(data[i][1])
  

  plt.title('Clustering using Expectation Maximization') 
  plt.xlabel('PC1') 
  plt.ylabel('PC2') 
  colorList = ['red','blue','green']
  for cluster in range(numOfCluster):
    plt.scatter(clusterX[cluster],clusterY[cluster],color=colorList[cluster])
  for cluster in range(numOfCluster):
    plt.scatter(meanData[cluster][0],meanData[cluster][1],color='black')
  #plt.show()
  plt.savefig("EM.png")


def EMAlgo(data,numOfCluster,dimension):
  # initialize mean,co-variance, phi
  random.seed(42)
  randomIndices = random.sample(range(0, len(data)-1), 3)
  print("Random indices :",randomIndices)
  meanData = np.array([data[randomIndices[0]],data[randomIndices[1]],data[randomIndices[2]]])
  print("Mean data shape : ",meanData.shape)
  print("Initial mean :",meanData)

  '''  
  meanValue = meanData.mean(axis=0)
  meanValue = meanValue.reshape(1,2)
  print(meanValue.shape)
  print(meanValue)
  '''

  covarianceData = np.zeros((numOfCluster,dimension,dimension))
  for cluster in range(numOfCluster):
    for val in range(len(data)):
      differnce = np.array(data[val]-meanData[cluster])
      submatrix = np.zeros((dimension,dimension))
      for valX in range(dimension):
        for valY in range(dimension):
          submatrix[valX][valY] = differnce[valX] * differnce[valY]
      covarianceData[cluster] += submatrix
    covarianceData[cluster] = covarianceData[cluster]/numOfCluster 
    
  print("Initial covariance :",covarianceData)

  #covarianceData = np.array([np.cov(data,rowvar=0)]*numOfCluster)
  #print("Covariance Data shape :",covarianceData.shape)
  weightData = np.array([1/numOfCluster]*numOfCluster)
  print("Weight shape :",weightData.shape)

  
  # initialize log likelihood
  prevLogLiklihood = calculateLogLikelihood(data,numOfCluster,dimension,meanData,covarianceData,weightData)
  logliklihood = prevLogLiklihood

  threshold = 0.000001
  #threshold = 0.001
  iterationNum = 1

  while True:
    # E-step
    probabilityData = calculateEstep(data,numOfCluster,dimension,meanData,covarianceData,weightData)
    #print("Probability shape :",probabilityData.shape)

    # M-step
    meanData,covarianceData,weightData = calculateMStep(data,numOfCluster,dimension,meanData,covarianceData,weightData,probabilityData)
    #print("updated mean shape :",meanData.shape)
    #print("updated covariance shape :",covarianceData.shape)
    #print("updated weight shape :",weightData.shape)
    #print(covarianceData)
    prevLogLiklihood = logliklihood
    logliklihood = calculateLogLikelihood(data,numOfCluster,dimension,meanData,covarianceData,weightData)

    if abs(logliklihood-prevLogLiklihood)<threshold and logliklihood > -np.inf:
      print("Done")
      return meanData,covarianceData,weightData,iterationNum
    iterationNum += 1

    if iterationNum % 10 == 0:
        print("Iteration " + str(iterationNum)+ " done !!")

# PCA
dimensionX,dimensionY,convertedInput = PCATest()

# EM Algorithm
numOfCluster = 3
meanData,covarianceData,weightData,iterationNum = EMAlgo(convertedInput,3,2)
print("Mean")
print(meanData)
print("covariance")
print(covarianceData)
print("weight")
print(weightData)
print("iteration Number",iterationNum)
# plot EM
plotEMAlgo(convertedInput,numOfCluster,2,meanData,covarianceData,weightData)