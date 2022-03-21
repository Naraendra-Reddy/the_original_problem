import numpy as np
import cv2
from tqdm import tqdm
import time
#import pdb

alpha=0.3
no_of_bins=60
angle_max=180
angle_min=-180
bins = np.linspace(angle_min,angle_max,no_of_bins)
#-360 and 360 works as well

def train(path_to_images = 'data/training/images', 
          csv_file = 'data/training/steering_angles.csv'):
    '''
    First method you need to complete. 
    Args: 
    path_to_images = path to jpg image files
    csv_file = path and filename to csv file containing frame numbers and steering angles. 
    Returns: 
    NN = Trained Neural Network object 
    '''
    data = np.genfromtxt(csv_file, delimiter = ',')
    frame_nums = data[:,0]
    steering_angles = data[:,1]  
    #X- storing all images
    X = []
    for i in range(1500):     
        im_full = cv2.imread(path_to_images + '/' + str(int(i)).zfill(4) + '.jpg')
        im_full= img_proc(im_full)
        X.append(np.ravel(np.array(im_full)))
     #print('executed x part')   
    X = np.reshape(X,(1500,784))
    #Y           
     #values that of diffent points on a normal distribution from 0 to 1
     #Multi class classifie, each class is a bucket, partition y's into these buckets
    Y=np.zeros((len(data),no_of_bins)) 
    arr = [0.2, 0.33, 0.66, 0.90,1]  #0.25,0.50,0.75,0.9,1
    vals= np.linspace(0,no_of_bins-1,no_of_bins)   
    for i,angles in enumerate(np.matrix(steering_angles).transpose()):
        pos= int(np.interp(angles,bins, vals)) 
        #check here
        if (((pos-4)>=0 and (pos+4)<= 63)):
            Y[i][pos-4: pos+1] += arr 
            Y[i][pos+1:pos+5] = arr[::-1][1:5]
          #pdb.set_trace()   
  
  
    # Train your network here. You'll probably need some weights and gradients!
    NN = Neural_Network()
    #grads = NN.computeGradients(X, y)
    J=[]
    for i in range(2000):
      #grads = NN.computeGradients(X, y)
      grads=NN.computeGradients(X,Y)
      #W1 and W2
      params=NN.getParams()
      params[:]=params[:]-((alpha*grads[:])/(len(X)))  #Check here 
      NN.setParams(params) #updating weights
      J.append(NN.costFunction(X,Y))    
    #pdb.set_trace()
    return NN


def predict(NN, image_file):
    '''
    Second method you need to complete. 
    Given an image filename, load image, make and return predicted steering angle in degrees. 
    '''
    
    im_full = cv2.imread(image_file)
    reshaped_img=img_proc(im_full)
    flattened_img= np.ravel(np.array(reshaped_img))
    yhat=np.argmax(NN.forward(flattened_img))
    pred_value=bins[yhat]    
    return pred_value

def img_proc(image):
  gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
  resized_img=cv2.resize(gray, (28,28))
  #blur=cv2.GaussianBlur(gray,(5,5),0)
  #cv2.Canny(blur, 50,150)
  return resized_img /255.0 
#Regularization
class Neural_Network(object):
    def __init__(self, Lambda=0):        
        #Define Hyperparameters, according to empirical data,ip=784, h1=128,h2=64 are ideal values
        self.inputLayerSize = 784
        self.outputLayerSize = 60
        self.hiddenLayerSize = 128
        
        #Weights (parameters) np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)*np.sqrt(1./self.inputLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)*np.sqrt(1./self.hiddenLayerSize )
        
        #Regularization Parameter:
        self.Lambda = Lambda
        
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        #Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3) + self.Lambda*self.W2
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        #Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2) + self.Lambda*self.W1
        
        return dJdW1, dJdW2
    
    #Helper functions for interacting with other methods/classes
    def getParams(self):
        #Get W1 and W2 Rolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single parameter vector:
        W1_start = 0
        W1_end = self.hiddenLayerSize*self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], \
                             (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], \
                             (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))