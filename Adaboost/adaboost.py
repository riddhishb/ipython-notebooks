
import numpy as np
import matplotlib.pyplot as plt


def weaklearner(thr,sign,dim,x):
    if(sign == 1):
        y = (x[:,dim] >= thr)
    else:
        y = (x[:,dim] < thr)

    y = y.astype(np.int64)
    y[np.where(y==0)] = -1
    return y

print "Generating Simulated Data"
# Code to enter the values of these variables
T = 10
N = 1000
dim = 2
    
x = np.random.randn(N, 2)  # dim=2

s = (N, 1)
# label = np.zeros(s) #linear separation example
label = np.zeros(s)  # nonlinear separation example
# for index in range(0,N):
#label[index] = x[index][0] < x[index][1]
for index in range(0, N):
    label[index] = (x[index][0]**2 + x[index][1]**2) < 1

label = label * 1.0

pos1 = np.nonzero(label == 1)
pos2 = np.where(label == 0)[0]
label[pos2] = -1

# plots the data

plt.figure()
plt.plot(x[pos1, 0], x[pos1, 1], 'b*')
plt.plot(x[pos2, 0], x[pos2, 1], 'r*')
plt.axis([-3, 3, -3, 3])
plt.legend('class 1', 'class 2', loc=2)
plt.title("Simulated (Original) data")

# declare parameters
weight = np.ones(N, dtype = np.float64) / (N)
err = np.ones(T, dtype = np.float64) * np.inf
alpha = np.zeros(T, dtype = np.float64)
h = np.zeros([T,3], dtype = np.float64)
thresholds = np.arange(-3.0, 3.0, 0.1)

print "Training"
for t in range(T):
    for thr in thresholds:
        for sign in [-1, 1]:
            for dim in [0, 1]:
                tmpe = np.sum(weight * (weaklearner(thr,sign,dim,x) != label[:,0]).astype(np.int64))
                if( tmpe < err[t]):
                    err[t] = tmpe
                    h[t,0] = thr
                    h[t,1] = sign
                    h[t,2] = dim
   
    if(err[t] >= 0.5): 
        print "error"
        break

    alpha[t] = 0.5 * np.log((1-err[t])/err[t])
    # % we update D so that wrongly classified samples will have more weight
    weight = weight * np.exp(-alpha[t] * label * weaklearner(h[t,0],h[t,1],h[t,2],x));
    weight = weight / np.sum(weight);


finalLabel = np.zeros_like(label);
misshits = np.zeros(T)

print "Testing"

for t in range(T):

    finalLabel = finalLabel + alpha[t] * weaklearner(h[t,0],h[t,1],h[t,2],x);
    tfinalLabel = np.sign(finalLabel);
    misshits[t] = np.sum((tfinalLabel != label[:,0]).astype(np.int64))/N;    
    
    pos1 = np.where(tfinalLabel == 1);
    pos2 = np.where(tfinalLabel == -1);


print "Results"
plt.figure()
plt.plot(x[pos1, 0], x[pos1, 1], 'b*')
plt.plot(x[pos2, 0], x[pos2, 1], 'r*')
plt.axis([-3, 3, -3, 3])
plt.legend('class 1', 'class 2', loc=2)
plt.title("Tested (Original) data")

# % plot miss hits when more and more weak learners are used
plt.figure()
plt.plot(misshits)
plt.ylabel('miss hists')
plt.show()