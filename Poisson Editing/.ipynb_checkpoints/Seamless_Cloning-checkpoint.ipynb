{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This is an [jupyter](http://jupyter.org) notebook.\n",
    "Lectures about Python, useful both for beginners and experts, can be found at http://scipy-lectures.github.io.\n",
    "\n",
    "Open the notebook by (1) copying this file into a directory, (2) in that directory typing \n",
    "jupyter-notebook\n",
    "and (3) selecting the notebook."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# <font color= 'blue'> Poisson Image Editing</font>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Seamless Image Cloning\n",
    "\n",
    "***\n",
    " A notebook by ***Dhruv Ilesh Shah***\n",
    "***\n",
    "\n",
    "In this exercise, we will learn and implement Seamless Image Cloning by employing the Poisson Solver.\n",
    "\n",
    "***Packages Used:*** Python 2.7, Numpy, Matplotlib, openCV 3.1.0,  `gimp 2.8` (recommended)\n",
    "\n",
    "*openCV has been used only for Image importing purposes, and can be replaced by PIL as well. Similarly, Matplotlib has been used only for displaying results inline.*\n",
    "***"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Motivation\n",
    "\n",
    "Image cloning and morphing have been some of the most common purposes of Image Processing and editing. Most of us have used these tools at a very abstract level through packages like Photoshop or GIMP etc. But have you ever thought about what goes into actuating this process? Here we explain the math behind the idea and the algorithm for seamlessly cloning an image *(or a portion thereof)* onto another image. The dictionary defines ***seamless*** as *smooth and without seams or obvious joins* and hence our objective would be to make an image as smooth and natural as possible.\n",
    "\n",
    "Soon after, we'll get our hands dirty by actually coding the algorithm and running it on a sample image to achieve the results for oursef. Towards the end, we will also look at the various parameters that can be tweaked to achieve better results.\n",
    "***"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Understanding Seamless Cloning\n",
    "\n",
    "Say you want to clone/morph a portion of image 1 onto a position on image 2. The simplest way to start off is to simply **paste** that portion of image on the other. But well, as you'd expect, the border has a very *sharp* change in features (known as gradient) and hence we can easily claim that the image is forged. What we need is a way to create a softer gradient, so that the image looks more natural.\n",
    "\n",
    "![Seamless Cloning Example](images/seamless_cloning_ex.jpg)\n",
    "<center> Courtesy: Vision @ Berkeley</center>\n",
    "\n",
    "This image set represents one of the best output I've seen this algorithm generate. As you see above, the softening of edges can be employed to make the image look better and the texture of the cloned image is preserved giving an almost natural looking eagle in the blue sky (*Mesmerising, isn't it?*)! This is a common blendng style in many Image Editing tools. Our goal, in this notebook, is to implement a function that can do a similar task from scratch. *(Do note that a similar function already exists in openCV)*\n",
    "\n",
    "At a very naive level, imagine the problem as solving the Laplace's Equation in 2 dimensions. The boundary of the cloned portion refer to the Boundary Conditions and you have to find a smooth curve fitting the conditions, such that no local extrema exist within the area, giving you a smooth image. This, at a basic level, is how the algorithm is evaluated. Now, let's get into the math behind the algorithm, and discretisation of the algorithm!\n",
    "***"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### The Math behind the Problem\n",
    "\n",
    "It was well known to psychologists [Land and McCann 1971] that slow gradients of intensity, which are suppressed by the Laplacian operator, can be superimposed on an image with barely noticeable effect. Conversely, the second-order variations extracted by the Laplacian operator are the most significant perceptually.\n",
    "\n",
    "Secondly, a *scalar* function on a bounded domain is uniquely defined by its values on the boundary and its Laplacian in the interior. The Poisson equation therefore has a unique solution and this leads to a sound algorithm.\n",
    "\n",
    "So, given methods for crafting the Laplacian of an unknown function over some domain, and its boundary conditions, the Poisson equation can be solved numerically to achieve seamless filling of that domain. This can be replicated independently in each of the channels of a color image. Solving the Poisson equation also has an alternative interpretation as a minimization problem: it computes the function whose gradient is the closest, in the L2 -norm (*Google!*), to some prescribed vector field — the *guidance vector field* — under given boundary conditions. In that way, the reconstructed function interpolates the boundary conditions inwards, while following the spatial variations of the guidance field as closely as possible."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Poisson Solution to *Guided Interpolation*: ** Interpolation, in the context of images, refers to being able to predict, and assign, values to an undefined pixel, given the surrounding values. It is one of the simplest ways to smoothen out an image and make it seamless. A good explanation on the topic can be found [here](https://prieuredesion.github.io/image%20processing/2016/05/15/interpolation.html). ![Guided Interpolation](images/guided_interpolation.png)\n",
    "\n",
    "The above image gives a great idea of the task at hand. Note the terminologies, as it would be used later as well. ***g*** refers to the portion of the source image that goes ON the $\\Omega$ portion of the destination image. ***v*** is the gradient field that defines features like texture and gradience in that portion. ***f*** is the symbol used to denote the cloned portion that will be processed.\n",
    "\n",
    "The simplest interpolant f over Ω is the membrane interpolant defined as the solution of the minimization problem:\n",
    "\n",
    "$$\\min_f \\int_{\\infty}^{} |\\nabla f|^2 \\hspace{0.5cm} with \\hspace{0.5cm} f|_{d\\Omega} = f^*|_{d\\Omega} $$\n",
    "where $\\nabla$ is the gradient operator. The minimisation problem mentioned above is more like solving the Laplace's equation ($\\nabla^2 f = 0$) with no gradient field. In the presence of a gradient vector field, we define the problem as follows:\n",
    "$$\\min_f \\int_{\\infty}^{} |\\nabla f - v|^2 \\hspace{5mm} with \\hspace{5mm} f|_{d\\Omega} = f^*|_{d\\Omega} $$\n",
    "\n",
    "The nature of this *guidance field* defines the nature of cloning, and various other results can be obtained by only tweaking this ***v*** from this algorithm, keeping everything else the same. The algorithm works by solving the *Dirichlet Boundary Condition* in the region $d\\Omega$, which is the boundary of the region $\\Omega$. The algorithm used for solving is known as the ***Jacobi's Iterative Method***. \n",
    "\n",
    "*Understanding the complete math may require deep knowledge of PDEs and related problems, and it is fine if you can accept the concept intuitively. The next part, then, would make more sense and illustrate how all the math is implemented*."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Employing the *Discrete* Poisson Solver\n",
    "\n",
    "As you would already be wondering, solving an integral and minimising an expression directly is not very intuitive, for implementing in a code. Instead, we choose to *approximate* and discretise the expressions so that they make more sense.\n",
    "\n",
    "For discrete images the problem can be discretized naturally using the underlying discrete pixel grid. Without loss of generality, we will keep the same notations for the continuous objects and their discrete counterparts: S, $\\Omega$ now become finite point sets defined on an infinite discrete grid. Note that S can include all the pixels of an image or only a subset of them. For each pixel p in S, let $N_p$ be the set of its 4-connected neighbors which are in S, and let p, q denote a pixel pair such that q ∈ $N_p$. The boundary of Ω is now $d\\Omega$ = {p ∈ S \\ $\\Omega : N_p \\cap \\Omega \\neq \\emptyset$}. Let $f_p$ be the value of f at p. The task is to compute the set of intensities $f|_\\Omega = f_p, p \\in \\Omega $.\n",
    "\n",
    "For Dirichlet Boundary Conditions, the finite difference discretisation yields the following quadratic optimisation problem.\n",
    "\n",
    "$$ \\min_{f|_\\Omega} \\sum_{\\langle p,q\\rangle\\cap\\Omega\\neq\\emptyset} (f_p-f_q-v_{pq})^2 \\hspace{3mm} with \\hspace{3mm} f_p=f_p^* \\hspace{3mm} for\\:all\\:p \\in d\\Omega$$\n",
    "\n",
    "Where $v_{pq}$ is the projection of $v(\\frac{p+q}{2})$  on the oriented edge $[p,q]$ as seen in the continuous domain. To convert this in the discrete domain, we can think of it as the gradient field along the line joining pixels $p$ and $q$. It's solution satisfies the following simultaneous linear equations:\n",
    "\n",
    "$$for\\:all\\:p \\in \\Omega,\\,|N_p|f_p-\\sum_{q\\in N_p\\cap\\Omega}f_q\\,=\\,\\sum_{q\\in N_p\\cap d\\Omega}f_q^* + \\sum_{q\\in N_p}v_{pq}$$\n",
    "\n",
    "Here, $|N_p|$ refers to the number of pixels in the neighbourhood of $p$ that also lie in $\\Omega$, ie, the cardinality. The method of obtaining this relation from the above is non-trivial and requires significant background in Calculus and Differential Equations. For beginners, assume the above is given and try to tackle the problem intuitively. The equations above form a classical, [positive-definite system](http://www.math.usm.edu/lambers/mat610/sum10/lecture7.pdf). Since the boundary $d\\Omega$ can be arbitrary, we must use well-known iterative solvers like the Gauss-Siedel iteration with successive overrelaxation.\n",
    "***"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Let's tackle the problem in Layman's terms\n",
    "\n",
    "This section is for you if (1) you have a slight idea of what is happening till now, and wish to understand further by using intuition, and (2) you have *no clue whatsoever* and wish to understand the algorithm.\n",
    "\n",
    "Let us begin by simple Laplacian Solvers, ie, $\\nabla^2 = 0$. skipping all the underlying math, you have probably been told that the function obtained does not have any maxima/minima in its interior and is a completely smooth shape. Converting this to the domain of *images*, let's look at the value of the function $f$ as it's pixel value in [0,255]. Clearly, given a boundary condition, the unique solution to Laplace's equation will result in smoothening of the interior and no sharp edges. This is what a boundary should look like if it is *seamless*.\n",
    "\n",
    "*But wait!* You also know that given a B.C., Laplace's equation has a unique solution, and given enough time (read: iterations) it would be independent of the cloned image! *Alas!* Now what? How do we preserve the features of the original image, which are in fact more important that smoothening (what use is seamless, if there's no cloning :p ). Here's where the idea of a gradient field comes in. It preserves the gradient, ie, texture and modulation in features of the image. The real math was stated above, but intuitively, it decides the shape and texture of the smooth curve fitting in the region. The Laplace's Solver is now replaced by the Poisson Solver, and the gradient field drives the interior of the region. Depending on the gradient field, certain or all features of the cloned portion can be preserved and or highlighted.\n",
    "\n",
    "The final outcome is decided by the choice of this gradient field. Next up is the simplest and optimal gradient field for Seamless Cloning. Towards the end, we will talk of other applications that can be achieved by changing this gradient field.\n",
    "\n",
    "***"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Implementing Seamless Cloning\n",
    "\n",
    "From the algorithm discussed in the earlier part, we see that the only missing information is the *gradient field **v***. The gradient field defines the features and textures of the image that we wish to introduce. Intuitively, what do you think would be an ideal gradient field for the region $\\Omega$ in the current situation?\n",
    "\n",
    "We have the source image portion in the region $\\Omega$ and we wish to blend it with the destination image. An ideal choice of gradient field would be the gradient vector field of the source image itself! This way, we preserve the features of the portion that we want as the dominant feature in $\\Omega$. We write this as:\n",
    "$$ v = \\nabla g $$\n",
    "Thus, the governing conditions now become\n",
    "\n",
    "$$ \\Delta f = \\Delta g\\,\\, over\\:\\:\\Omega,\\,with f|_{d\\Omega}=f|_{d\\Omega}^*$$\n",
    "\n",
    "For the numerical (discrete) counterpart of the definition of gradient field, we have:\n",
    "\n",
    "$$ v_{pq}=g_p-g_q,\\:\\, for\\:all\\:\\langle p,q \\rangle $$\n",
    "\n",
    "\n",
    "The seamless cloning tool thus obtained ensures the compliance of source and destination boundaries. It can be used to conceal undesirable image features or to insert new elements in an image, but with much more flexibility and ease than with conventional cloning. We will now implement this algorithm on Python.\n",
    "***"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## <font color=\"blue\"> Let's get Coding!!! </font>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Firstly, we import all necessary packages. All the packages can be downloaded using `pip`. As an alternative to openCV, you can use PIL as well. Refer to the [readme](README.md) for more details."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "%matplotlib inline \n",
    "\n",
    "import cv2\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For applying the algorithm, you will need the cloned image and a mask that identifies the cloned portion of the image. The purpose of the mask will become clear in the subsequent steps.  \n",
    "***How to obtain the mask?*** *This method needs improvement - suggestions welcome!*\n",
    "* Given that you clone the image on your system, I recommend you use `gimp` for the purpose. Open `GIMP Image Editor`.\n",
    "* Open the image on which cloning must occur, the destination image.\n",
    "* In another window, open the source, whose portion is to be cloned. Using the select tool, select and copy the portion.\n",
    "* Paste the portion on the destination image. You will see that a new layer with the pasted portion will be created. Using the `Transform` tool, move the portion to desired location and scale accordingly, using the `Scale` tool.\n",
    "* Export this image and save your cloned file as a `jpeg` or `png` file. Now we must make the mask.\n",
    "* Create a new `background layer` and place it below the pasted layer. Also, hide the destination image layer. Select the bg-layer and color it black using the Color Tool.\n",
    "* You will now have the cloned portion on a black background. Using the Thresholding tool (or on `cv2`), threshold all pixels above 1 as 255.(*This should work unless your image has a pixel value 0, in which case you may have to use floodFill or explicitly modify the mask file*).\n",
    "* Export this file as the mask image in the `jpeg` or `png` format.\n",
    "\n",
    "Let's now begin with the algorithm implementation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "im_cloned = cv2.imread(\"stinkbug_cloned.png\",  cv2.IMREAD_GRAYSCALE)\n",
    "im_mask = cv2.imread(\"stinkbug_cloned_mask.png\", cv2.IMREAD_GRAYSCALE)\n",
    "it = 200; # Set number of iterations"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "After importing the image and mask, we create a temporary image to store the current value of pixels as the new ones are being calculated. This is because we want to simultaneously update values. * The iterative method used is called the ** Jacobi's Method. ** * In contrast to this, the ***Gauss-Siedel method*** involves no temporary matrix and the update occurs along with the code. This converges faster, but may result in loss of information and unexpected outcomes in some cases."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n",
      "50\n",
      "100\n",
      "150\n"
     ]
    }
   ],
   "source": [
    "im_temp = im_cloned.copy()\n",
    "im_seamless = im_temp.copy()\n",
    "sigma = []\n",
    "for i in range(im_cloned.shape[0]):\n",
    "        for j in range(im_cloned.shape[1]):\n",
    "            if (im_mask[i,j]==255):\n",
    "                sigma.append([i,j])\n",
    "\n",
    "for a in range(it):\n",
    "    for [i,j] in sigma:\n",
    "        term = 10000\n",
    "        term = term + im_seamless[i+1, j]+im_seamless[i-1, j]+im_seamless[i, j-1]+im_seamless[i, j+1]\n",
    "        if(im_mask[i-1, j]==255):\n",
    "            term = term + im_cloned[i,j]-im_cloned[i-1,j]\n",
    "        if(im_mask[i+1, j]==255):\n",
    "            term = term + im_cloned[i,j]-im_cloned[i+1,j]\n",
    "        if(im_mask[i, j+1]==255):\n",
    "            term = term + im_cloned[i,j]-im_cloned[i,j+1]\n",
    "        if(im_mask[i, j-1]==255):\n",
    "            term = term + im_cloned[i,j]-im_cloned[i,j-1]\n",
    "        im_temp[i,j] = (term-10000)/4\n",
    "    im_seamless = im_temp.copy()\n",
    "    if a%50==0:\n",
    "        print a"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As is visible from the code, we scan the whole cloned image for points where the corresponding value of the mask is 255 (*white*), and update `im_temp` as suggested in the algorithm. The gradient for the source image is defined as zero at the boundary points.\n",
    "*Note that the pixel values in cv2 are of `ubyte_scalar` type and hence don't support addition due to overflow. `10000` is a random large number that helps maintain the variable `term` as a normal integer and hence allow addition without overflow*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.text.Text at 0x7f8fd5619510>"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\nt261X09XeRRlTaP1M+241lloTrh2sridsKe2l3rImuqpqgOKkwxN/UXmW+bvwgsvJBKJcN9999nP\n8q09mTwe3ZSGkww2bdrEU089xec//3ntd6xNZcjk9XopLy/nhRde4ODBg7YlLgwQcS9+4XCY1atX\ns3LlStubFQuwVVVVtqIvLi6msrLS9gjE7h6TrppmbE/LN2izEz+n55No06ZNNgjVo2AhV3E6fcBE\nTTOpcAdL3mTdy9alnLduUNIpU52FJYNOPB8fH+fb3/42X/jCF7j55pu1L1fI6Z0sN+HOmyw9cd3X\n18e///u/c8sttxgVjEpy3dRX6s8++2wuv/xyPvWpT1FVVWUfVyziBwIBioqKOP300+3yxsbGCIVC\nJBIJfv7zn9PV1UU0GuUjH/kIg4ODLFq0iHQ6zc9+9jO+8pWv2Ecm63jNZq1X0vft2zetHYNZwrUO\nk4UcwaCGg36txWRh63As38v5ynmo00hikPn85z/Pddddx7Zt21i9evUkPoQcdAaWeG6yvtU4Ip8f\n/vCHLF68mEsvvZRsNmtPCary0mFI/MufCfzHf/xHHn30UZqbm4lEIiSTSRoaGnC5rA+biI+Lv+Md\n7yCdTuPxeOx5fp/Px44dO+jo6GDbtm32V64qKyspKirinnvu4frrr8/x7Gca20eq7E3HvXYCi6R4\njRNhk+htb3vbpPk3nTKVrUVButHPydqW81OVpww6tWwd8E2jrQlApnjiPxaLcdlll/Gtb32La6+9\ndhJvsjeh6xT5ypMHrn379vGnP/2Jm266aZJbq/N+1DDTyY1f/vKX2bJlC+eeey7JZJJDhw6xcKFl\n+G7atImXXnqJ5cuXU1tbi8/no6uri82bN9PZ2cl1111Hc3Ozvf2srq6O/v5+EokE8+fPt3c3yOWp\ndV2+fDnNzc12+ObNm7WyKICOOa5l0uHMZBHnMzx0A4AqN9WLNA0aOlyoZavleDwebr31Vr74xS+y\nYsUK+4hq0+Dh1Id1JPcJt9vN888/z3nnnceSJUscByynPqzqmGAwSGVlJX19fbb3OTg4SENDA0VF\nRfzgBz8gHo/T2tpqT0Pu2bOHrVu3Ul9fz/ve9z6KiopYsWKFnV8mk2Hfvn1ccMEF9icPnSz5acT2\nESt7cdzrt8k97vUXwE+Af8Vyc1s4/MFmIxXingrg64Dh5Po4KWdZ0Tt1GB1IdECSy9Txr5IAa1NT\n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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f8fd77d8e50>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plt.subplots(1, 2)\n",
    "ax[0].imshow(im_cloned,cmap = 'gray')\n",
    "ax[0].set_title('Normal Cloning')\n",
    "ax[1].imshow(im_seamless, cmap='gray')\n",
    "ax[1].set_title('Seamless Cloning')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "As you can see, the images have blended pretty well, with features of source image preserved, and the cloning almost seamless! To view the image in a separate window, execute the next cell.\n",
    "\n",
    "**Note: ** The above implementation has been made only in one channel (*grayscale*), but running the same algorithm in the three different channels will give similar results for RGB/HSV/CIE-Lab schemes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "cv2.imshow(\"Naive Cloning\", im_cloned)\n",
    "cv2.imshow(\"Seamless Cloning\", im_seamless)\n",
    "cv2.waitKey(0); cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### What Comes Next?\n",
    "\n",
    "Having understood the algorithm and implementation, it shouldn't be tough to experiment with the parameter **it** and identify an optimal figure depending on the size of $\\Omega$. Also, you can try vectorising some potions which haven't been done so yet and improve runtime.\n",
    "\n",
    "Also, by gently changing the *gradient field* we can obtain several desirable effects - Mixing Gradients, Feature Exchange, Inserting Transparent Objects, Texture Flattening - the list is endless. Google is your friend here, and so is [this paper](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf). Good luck exploring the topic! There's so much more to do... All you'd need is the will to do it!\n",
    "\n",
    "*Cheers!*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Thank You!\n",
    "***"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
