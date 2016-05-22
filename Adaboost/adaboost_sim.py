"""
All matrices used are implemented via numpy.

The following variables are used:
-> N: The number of samples or data-points.
-> T: The number of iterations in our boosting algorithm.
-> dim: The number of parameters recorded for each data-point.
		(for an image we can choose R,G and B intensities as features and then dim=3)
-> x: The data. It is an N x dim matrix. 
-> label: N x 1 array that stores the known labels for each data-point.
-> final_label: N x 1 array that stores the labels generated for each data-point by the final strong classifier.
-> weight: N x 1 array that stores the weight for each data-point.
-> h: T x 3 array that stores the weak classifiers selected after each iteration:
	   h[index][0]= threshold
	   h[index][1]= dim (data dimension)
	   h[index][2]= pos (the sign of the classifier, +1/-1)
-> alpha: T x 1 array that stores the weight of each weak classifier chosen to make up the final classifier.

"""
#import math
import numpy as np
import matplotlib.pyplot as plt

T=10
dim=2
N =1000

def weakClassifier_error(i,j,k,x,weight,label): #Returns the error on the data for a given distribution of weights
								 #Takes the threshold, dimension of data and the 'sign' as input

	temp_err= np.float64(0) #The output of this function
	temp= np.int64(0)
	y=np.zeros(N, dtype = np.int64) #Initialise actual and expected labels to a perfect match( 0 = match , 1 = not a match)
	
	for p in range(N):
          temp = np.sign(x[p][j] - i)
          if(k == -1):      #Note that k is the 'sign', i is threshold, j is the dimension
			temp = -temp

          if(temp != label[p]): 
			temp_err = np.float64(temp_err + weight[p])
			y[p]=1  #This indicates a mismatch between known and generated label at this position. 
			        #To be used later in reassigning the weights.

	return [temp_err,y]

#x=label=weight=N=dim=T=None
#Code to enter the values of these variables
x = np.random.randn(N,2) #dim=2

label = np.zeros(N, dtype= np.int64) 
#for index in range(0,N): #linear separation example
	#label[index] = x[index][0] < x[index][1]

for index in range(0,N):   #nonlinear separation example
	label[index] = (x[index][0]**2 + x[index][1]**2) < 1
	
label = label * 1.0

pos1 = np.nonzero(label == 1)
pos2 = np.where(label==0)
label[pos2] = -1

#plots the data

plt.figure()
plt.plot(x[pos1,0], x[pos1,1], 'b*')
plt.plot(x[pos2,0], x[pos2,1], 'r*')
plt.axis([-3,3,-3,3])
plt.title("Original data")

#Actual program begins

#h and alpha together completely specify the final strong classifier 
h=np.zeros([T,3], dtype = np.float64) 
alpha=np.zeros(T, dtype = np.float64)

threshold=np.arange(-3.0,3.0,0.1) #This was the range in the MATLAB example. Can easily be changed to whatever.

weight = np.ones(N, dtype = np.float64) / (N)

y=np.zeros(N, dtype = np.int64) #To be used in the following function. If the assigned label does NOT match the label of the weak 
#classifier, for say the i'th data point, then a 'flag' is raised by setting y[i] to 1.

err  = np.ones(T, dtype = np.float64) * np.inf

for t in range(T):
	for i in threshold:
		for j in range(dim):
			for k in [-1,1]:
				[tmpe,y] = weakClassifier_error(i,j,k,x,weight,label)
				if(tmpe < err[t]): #storing the better classifier in h
					err[t] = tmpe
					y0 = y
					h[t][0]=i
					h[t][1]=j
					h[t][2]=k	 		  
						 
	if(err[t] > 0.5): 
		T= t
		print t, "Error!"    #We have run out of weak classifiers! So truncate the no: of iterations used
		break  

	alpha[t] = 0.5 * np.log((1.0-err[t])/err[t])
	

	for i in range(N):          #Reassign weigths for next iteration
		
		if(y0[i]==1):
			weight[i] = np.float64(weight[i]* np.exp(alpha[t]))
		else:
			weight[i] = np.float64(weight[i]* np.exp(-alpha[t]))

	weight = weight / np.sum(weight)


temp_sum = np.float64(0)
temp = np.int64(0)
final_label = np.zeros(N,dtype = np.float64)
for i in range(N):
	temp_sum = np.float64(0)
	for j in range(T):
          z=np.sign(x[i][ h[j][1] ] - h[j][0])
          if(h[j][2] == 1):	     
			temp = z
          else:
			temp= -z
          temp_sum= float(temp_sum + alpha[j]*temp)

	final_label[i] = np.sign(temp_sum)

		

#Now plot the generated labels	
pos1 = np.where(final_label == 1)
pos2 = np.where(final_label== -1)

plt.figure()
plt.plot(x[pos1,0], x[pos1,1], 'b*')
plt.plot(x[pos2,0], x[pos2,1], 'r*')
plt.axis([-3,3,-3,3])
plt.title("Generated data")
plt.show()




