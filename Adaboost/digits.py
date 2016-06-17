"""
All matrices used are implemented via numpy.
The following variables are used:
-> N: The number of samples or data-points.
-> T: The number of iterations in our boosting algorithm.
-> dim: The number of parameters recorded for each data-point.
        (for an image we can choose RGB intensities as features and then dim=3)
-> x: The data. It is an N x dim matrix.
-> label: N x 1 array that stores the known labels for each data-point.
-> final_label: Nx1 array that stores the labels generated for each data-point
                by the final strong classifier.
-> weight: Nx1 array that stores the weight for each data-point.
-> h: Tx3 array that stores the weak classifiers selected after each iteration:
       h[index][0]= threshold
       h[index][1]= dim (data dimension)
       h[index][2]= pos (the sign of the classifier, +1/-1)
-> alpha: T x 1 array that stores the weight of each weak classifier chosen to
            make up the final classifier.
-> final_alpha: Stores the weights for all the digits.
-> final_h: Stores the classifiers for all the digits.

"""
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
start_time = time.time()


with open('images_training.txt', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    x = list(reader)

x = np.array(x, dtype=np.float64)
# this array is of size 13x13000, for all the 1000 13x13 images 100 for
# each digit 0 to 9

T = 20
dim = 169
N = 1000

temp = np.zeros(N, dtype=np.int64)

# Returns error and calculated labels corresponding to


def weakClassifier_error(i, j, k, x, weight, label):
                                                # threshold i
                                                # dimension j
                                                # sign k on dataset x.
                                                # Original labels are stored in
                                                # label

    j_row = j / 13
    j_col = j % 13
    temp_err = np.float64(0)
    # Initialise actual and expected labels to a perfect match( 0 = match , 1
    # = not a match)
    y = np.zeros(N, dtype=np.int64)

    if(k == 1):
        temp = (x[j_row, j_col:13000:13] >= i)
    else:
        temp = (x[j_row, j_col:13000:13] < i)

    temp = np.int64(temp)
    temp[np.where(temp == 0)] = -1
    y = np.int64(temp != label)
    # Calculate error of this weak classifier on the weighted dataset
    temp_err = np.sum(y * weight)

    return [temp_err, y]


# Actual program begins
threshold = np.arange(0, 1.0, 0.05)
# h and alpha together completely specify the final strong classifier
final_alpha = np.zeros((10, T), dtype=np.float64)
final_h = np.zeros((10, T, 3), dtype=np.float64)

for p in range(10):
    h = np.zeros([T, 3], dtype=np.float64)
    alpha = np.zeros(T, dtype=np.float64)
    temp = np.zeros(N, dtype=np.int64)

    label = np.zeros(N, dtype=np.int64)
    label = label * 1.0
    label[p * 100: p * 100 + 100] = 1
    label[np.where(label == 0)] = -1

    weight = np.ones(N, dtype=np.float64) / (N)  # Initialise weights

    # Initially set error to infinity, to allow comparing with error of
    # classifiers
    err = np.ones(T, dtype=np.float64) * np.inf

    for t in range(T):
        for i in threshold:
            for j in range(dim):
                for k in [-1, 1]:
                    [tmpe, y] = weakClassifier_error(i, j, k, x, weight, label)
                    if(tmpe < err[t]):  # storing the better classifier in h
                        err[t] = tmpe
                        y0 = y
                        h[t][0] = i
                        h[t][1] = j
                        h[t][2] = k

        if(err[t] > 0.5):
            T = t
            # We have run out of weak classifiers! So truncate the no: of
            # iterations used
            print t, "Error!"
            break

        alpha[t] = 0.5 * np.log((1.0 - err[t]) / err[t])

        # y0=0 corresponded to correctly labelled datapoints. To reassign
        # weights,
        y0[np.where(y0 == 0)] = -1
        # we need -1 and not 0 at these positions

        weight = np.float64(weight * np.exp(alpha[t] * y0))  # Reassign weights
        weight = weight / np.sum(weight)  # Normalise reassigned weights

    final_alpha[p] = alpha
    final_h[p] = h


with open('images_training.txt', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    x = list(reader)

x = np.array(x, dtype=np.float64)

temp_sum = np.zeros((10, N), dtype=np.float64)
temp = np.zeros(N, dtype=np.float64)
final_label = np.zeros((10, N), dtype=np.float64)
misshits = np.zeros(T)

for p in range(10):
    label[100 * p: 100 * p + 100] = p

label = np.int64(label)
all_label = np.full(N, -1, dtype=np.int64)

for t in range(T):  # Calculate final labels
    for p in range(10):
        row = final_h[p][t][1] / 13
        col = final_h[p][t][1] % 13
        temp = final_h[p][t][2] * \
            np.sign(x[row, col: 13000: 13] - final_h[p][t][0])
        temp_sum[p] = np.float64(temp_sum[p] + final_alpha[p][t] * temp)
        final_label[p] = np.sign(temp_sum[p])
    for p in range(10):
        all_label[np.where(final_label[p] == 1)] = p
    misshits[t] = np.sum(np.float64(all_label != label)) / N

plt.figure()
plt.plot(misshits)
plt.ylabel('Miss hists')
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
