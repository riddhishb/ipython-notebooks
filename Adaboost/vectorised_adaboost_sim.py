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
import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()

T=20
dim=2
N =1000

temp = np.zeros(N, dtype= np.int64) 
def weakClassifier_error(i,j,k,x,weight,label): #Returns error and calculated labels corresponding to 
												#threshold i, dimension j, sign k on dataset x.
												#Original labels are stored in label 
 	
	temp_err= np.float64(0) 
	y=np.zeros(N, dtype = np.int64) #Initialise actual and expected labels to a perfect match( 0 = match , 1 = not a match)
	
	if(k == 1):
		temp = (x[:,j] >= i)
	else:
		temp = (x[:,j] < i)
		
         
	temp = np.int64(temp)
	temp[np.where(temp == 0)] = -1
	y = np.int64(temp != label)
	temp_err = np.sum(y*weight) #Calculate error of this weak classifier on the weighted dataset
          
	return [temp_err,y]


x = np.random.randn(N,2) #dim=2

label = np.zeros(N, dtype= np.int64) 

#label = x[:,0] < x[:,1]  #linear separation example
label = (x[:,0]**2 + x[:,1]**2) < 1   #nonlinear separation example
	
	
label = label * 1.0

pos1 = np.nonzero(label == 1)
pos2 = np.where(label==0)
label[pos2] = -1

#Plot the data

plt.figure()
plt.plot(x[pos1,0], x[pos1,1], 'b*')
plt.plot(x[pos2,0], x[pos2,1], 'r*')
plt.axis([-3,3,-3,3])
plt.title("Original data")

#Actual program begins

#h and alpha together completely specify the final strong classifier 
h=np.zeros([T,3], dtype = np.float64) 
alpha=np.zeros(T, dtype = np.float64)

threshold=np.arange(-3.0,3.0,0.1) 

weight = np.ones(N, dtype = np.float64) / (N) #Initialise weights

err  = np.ones(T, dtype = np.float64) * np.inf #Initially set error to infinity, to allow comparing with error of classifiers

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

	y0[np.where(y0==0)] = -1  #y0=0 corresponded to correctly labelled datapoints. To reassign weights, 
							  #we need -1 and not 0 at these positions

	weight = np.float64(weight* np.exp(alpha[t]*y0)) #Reassign weights

	weight = weight / np.sum(weight) #Normalise reassigned weights


temp_sum = np.zeros(N,dtype = np.float64)
temp = np.zeros(N,dtype = np.float64)
final_label = np.zeros(N,dtype = np.float64)
misshits = np.zeros(T)

for t in range(T):  #Calculate final labels
	temp= h[t][2]* np.sign(x[:, h[t][1] ] - h[t][0])
	temp_sum= np.float64(temp_sum + alpha[t]*temp)
	final_label = np.sign(temp_sum)
	misshits[t] = np.sum( np.float64(final_label != label) )/N 


		

#Now plot the generated labels	
pos1 = np.where(final_label == 1)
pos2 = np.where(final_label== -1)

plt.figure()
plt.plot(x[pos1,0], x[pos1,1], 'b*')
plt.plot(x[pos2,0], x[pos2,1], 'r*')
plt.axis([-3,3,-3,3])
plt.title("Generated data")
plt.show()

#Plot miss hits when more and more weak learners are used
plt.figure()
plt.plot(misshits)
plt.ylabel('Miss hists')
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))


