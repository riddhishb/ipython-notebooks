import numpy as np
import PIL
import PIL.Image as im
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import scipy.misc as spmi
import matplotlib.pyplot as plt

def possion_solver(dest_gs,source_gs, mask):
	tot_pixels = mask.size
	A = scipy.sparse.dok_matrix((tot_pixels,tot_pixels), dtype=np.float32)
	b = np.zeros(tot_pixels, dtype = np.float32)
	pix = mask.nonzero()
	y = pix[0][i]
    x = pix[1][i]
    
    # continue from here one
    # ye to ho jayega now




# Read images and make a naive clone
dest = im.open("images/dest.jpg")
dest = np.array(dest)

source = im.open("images/source.jpg")
source = np.array(source)

naive_clone = dest.copy()
mask = np.zeros((dest.shape[0], dest.shape[1]),dtype = np.int8)
mask[160:220,20:490] = 1

naive_clone[mask == 1] = source[mask==1]

plt.imshow(naive_clone)

# Now perform the possion editing for r,g,b channels
# plot the results

plt.show()