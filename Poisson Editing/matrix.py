import numpy as np
import PIL
import PIL.Image as im
import scipy.sparse as spr
import scipy.misc as spmi
import matplotlib.pyplot as plt
import cv2

def possion_solver(dest_gs,source_gs, mask):
  tot_pixels = mask.size
  A = spr.dok_matrix((tot_pixels,tot_pixels), dtype=np.float32)
  b = np.zeros(tot_pixels, dtype = np.float32)
  pix = mask.nonzero()
  mask1 = np.zeros((tot_pixels), dtype=np.float32)
  h = mask.shape[0]
  w = mask.shape[1]
  source_gs = source_gs.astype(np.float32)
  mode = 1 # Set to 2 if Mixed Gradients is desired

  sigma = []
  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
    	if (mask[i,j] != 0): sigma.append([i,j])
  for [i,j] in sigma:
    mask1[(i-1)*w + j]=1
    A[(i-1)*w +j, (i-1)*w +j]=4
    if (mask[i-1,j]>0): A[(i-1)*w+j,(i-2)*w+j]=-1
    else: b[(i-1)*w+j]= b[(i-1)*w +j] + dest_gs[i-1,j]

    if (mask[i+1,j]>0): A[(i-1)*w+j,(i)*w+j]=-1
    else: b[(i-1)*w+j]= b[(i-1)*w +j] + dest_gs[i+1,j]

    if (mask[i,j-1]>0): A[(i-1)*w+j,(i-1)*w+j-1]=-1
    else: b[(i-1)*w+j]= b[(i-1)*w +j] + dest_gs[i,j-1]

    if (mask[i,j+1]>0): A[(i-1)*w+j,(i-1)*w+j+1]=-1
    else: b[(i-1)*w+j]= b[(i-1)*w +j] + dest_gs[i-1,j+1]

    # Adding the gradient field
    vs = 4*source_gs[i,j]-(source_gs[i-1,j]+source_gs[i+1,j]+source_gs[i,j-1]+source_gs[i,j+1])
    b[(i-1)*w+j] = b[(i-1)*w+j] + vs

  mask1 = np.asarray([1 if mask1[i]>0 else 0 for i in range(mask1.size)]) # Vectorised :D
  b = np.asarray(b);
  b = b[np.where(mask1>0)[0]]
  print b.size
  count =0
  for i in range(mask1.size):
    if mask1[i]==0:
      A = np.delete(A,i-count,axis=0); A=np.delete(A,i-count,axis=1);
      count = count+1



 
  print A.shape #gives null here.. WHY!? Works fine elseways
  x = np.linalg.lstsq(A.T, b)[0];
  print x.shape








# Read images and make a naive clone
dest = cv2.imread("images/dest.jpg",0)
dest = np.array(dest)

source = cv2.imread("images/source.jpg",0)
source = np.array(source)

naive_clone = dest.copy()
mask = np.zeros((dest.shape[0], dest.shape[1]),dtype = np.int8)
mask[160:220,20:490] = 1

naive_clone[mask == 1] = source[mask==1]

# plt.imshow(naive_clone)
cv2.imshow("Here",naive_clone); cv2.waitKey(0); cv2.destroyAllWindows();
final_img = possion_solver(dest, source, mask);
# Now perform the possion editing for r,g,b channels?
# plot the results
# plt.imshow(final_img)
#cv2.imshow("Final", final_img); cv2.waitKey(0); cv2.destroyAllWindows();