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
import math
import numpy as np
import matplotlib.pyplot as plt

x=label=weight=N=dim=T=None
#Code to enter the values of these variables
T=10
dim=2
N =1000
x = np.random.randn(N,2) #dim=2

s = (N,1)
#label = np.zeros(s) #linear separation example
label = np.zeros(s) #nonlinear separation example
#for index in range(0,N):
	#label[index] = x[index][0] < x[index][1]
for index in range(0,N):
	label[index] = (x[index][0]**2 + x[index][1]**2) < 1
	
label = label * 1.0



pos1 = np.nonzero(label == 1)
pos2 = np.where(label==0)[0]
label[pos2] = -1

#plots the data

plt.plot(x[pos1,0], x[pos1,1], 'b*')
plt.plot(x[pos2,0], x[pos2,1], 'r*')
plt.axis([-3,3,-3,3])
plt.title("Original data")
plt.show()



#Actual program begins

#h and alpha together completely specify the final strong classifier 
h=np.zeros([T,3]) 
alpha=np.zeros(T)

threshold=np.arange(-3.0,3.0,0.1) #This was the range in the MATLAB example. Can easily be changed to whatever.

weight=np.zeros(N)
for i in range(N):
	weight[i]=1.0/N

y=np.zeros(N) #To be used in the following function. If the assigned label does NOT match the label of the weak 
              #classifier, for say the i'th data point, then a 'flag' is raised by setting y[i] to 1.


def weakClassifier_error(i,j,k): #Returns the error on the data for a given distribution of weights
								 #Takes the threshold, dimension of data and the 'sign' as input

	temp_err= 0.0 #The output of this function
	temp=0
	for p in range(N): #Initialise actual and expected labels to a perfect match( 0 = match , 1 = not a match)
		y[p]=0
	
	for p in range(N):
		if(k == 1):      #Note that k is the 'sign', i is threshold, j is the dimension
			temp =  int(x[p][j] >= i)
			if(temp==0): temp = -1
		else:
			temp =  int(x[p][j] < i)
			if(temp==0): temp = -1

		if(temp != label[p]): 
			temp_err = temp_err + weight[p]
			y[p]=1  #This indicates a mismatch between known and generated label at this position. 
			        #To be used later in reassigning the weights.

	return temp_err




for t in range(T):
	err=float("inf")
	for i in threshold:
		for j in range(dim):
			for k in [-1,1]:
				tmpe= weakClassifier_error(i,j,k)
				if(tmpe < err): #storing the better classifier in h
					err = tmpe
					h[t][0]=i
					h[t][1]=j
					h[t][2]=k	 		  
						 
						  

	if(err > 0.5): 
		T= t    #We have run out of weak classifiers! So truncate the no: of iterations used
		break  

	alpha[t] = 0.5 * math.log((1.0-err)/err)
	
	sum=0.0
	for i in range(N):          #Reassign weigths for next iteration
		
		if(y[i]==1):
			weight[i] = weight[i]* math.exp(alpha[t])
		else:
			weight[i] = weight[i]* math.exp(-alpha[t])

		sum=sum+weight[i]	

	weight = weight/sum
		



temp_sum = 0.0
temp = 0
final_label = np.zeros(N,int)
for i in range(N):
	for j in range(T):
		if(h[j][2] == 1):	     
			temp = int( x[i][ h[j][1] ] >= h[j][0] )
			if(temp==0): temp=-1
		else:
			temp = int( x[i][ h[j][1] ] < h[j][0] )
			if(temp==0): temp=-1
	
		temp_sum= temp_sum + alpha[j]*temp

	if(temp_sum >= 0):
		final_label[i] = 1
	else:
		final_label[i] = -1

		

#Now plot the generated labels	
pos1 = np.nonzero(final_label == 1)
pos2 = np.where(final_label== -1)

plt.plot(x[pos1,0], x[pos1,1], 'b*')
plt.plot(x[pos2,0], x[pos2,1], 'r*')
plt.axis([-3,3,-3,3])
plt.title("Generated data")
plt.show()




