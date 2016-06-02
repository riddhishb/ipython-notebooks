import numpy as np
import PIL
import PIL.Image as im
import scipy.sparse as spr
import scipy.misc as spmi
import matplotlib.pyplot as plt

def possion_solver(dest_gs,source_gs, mask):
  tot_pixels = mask.size
  A = scipy.sparse.dok_matrix((tot_pixels,tot_pixels), dtype=np.float32)
  b = np.zeros(tot_pixels, dtype = np.float32)
  pix = mask.nonzero()
  mask1 = np.zeros(1, tot_pixels)
  h = mask.shape[0]
  w = mask.shape[1]
  sigma = []
  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
    	if (mask[i,j] != 0) sigma.append([i,j])
  for [i,j] in sigma:
    mask1[(i-1)*w + j]=1
    A[(i-1)*w +j, (i-1)*w +j]=4
    if (mask[])

# Read images and make a naive clone
dest = im.open("images/dest.jpg",0)
dest = np.array(dest)

source = im.open("images/source.jpg",0)
source = np.array(source)

naive_clone = dest.copy()
mask = np.zeros((dest.shape[0], dest.shape[1]),dtype = np.int8)
mask[160:220,20:490] = 1

naive_clone[mask == 1] = source[mask==1] # 

plt.imshow(naive_clone)
plt.show()
final_img = possion_solver(dest, source, mask);
# Now perform the possion editing for r,g,b channels?
# plot the results
plt.imshow(final_img)
plt.show()