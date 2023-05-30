'''
@author: Vignesh Srinivasan
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c) 2016-2017, Vignesh Srinivasan, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''


import numpy as np
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable

import skimage.io
import skimage.feature
import skimage.filters

def vec2im(V, shape = () ):
    '''
    Transform an array V into a specified shape - or if no shape is given assume a square output format.

    Parameters
    ----------

    V : numpy.ndarray
        an array either representing a matrix or vector to be reshaped into an two-dimensional image

    shape : tuple or list
        optional. containing the shape information for the output array if not given, the output is assumed to be square

    Returns
    -------

    W : numpy.ndarray
        with W.shape = shape or W.shape = [np.sqrt(V.size)]*2

    '''
    
    if len(shape) < 2:
        shape = [np.sqrt(V.size)]*2
        shape = list(map(int, shape))
    return np.reshape(V, shape)


def enlarge_image(img, scaling = 3):
    '''
    Enlarges a given input matrix by replicating each pixel value scaling times in horizontal and vertical direction.

    Parameters
    ----------

    img : numpy.ndarray
        array of shape [H x W] OR [H x W x D]

    scaling : int
        positive integer value > 0

    Returns
    -------

    out : numpy.ndarray
        two-dimensional array of shape [scaling*H x scaling*W]
        OR
        three-dimensional array of shape [scaling*H x scaling*W x D]
        depending on the dimensionality of the input
    '''

    if scaling < 1 or not isinstance(scaling,int):
        print('scaling factor needs to be an int >= 1')

    if len(img.shape) == 2:
        H,W = img.shape

        out = np.zeros((scaling*H, scaling*W))
        for h in range(H):
            fh = scaling*h
            for w in range(W):
                fw = scaling*w
                out[fh:fh+scaling, fw:fw+scaling] = img[h,w]

    elif len(img.shape) == 3:
        H,W,D = img.shape

        out = np.zeros((scaling*H, scaling*W,D))
        for h in range(H):
            fh = scaling*h
            for w in range(W):
                fw = scaling*w
                out[fh:fh+scaling, fw:fw+scaling,:] = img[h,w,:]

    return out


def repaint_corner_pixels(rgbimg, scaling = 3):
    '''
    DEPRECATED/OBSOLETE.

    Recolors the top left and bottom right pixel (groups) with the average rgb value of its three neighboring pixel (groups).
    The recoloring visually masks the opposing pixel values which are a product of stabilizing the scaling.
    Assumes those image ares will pretty much never show evidence.

    Parameters
    ----------

    rgbimg : numpy.ndarray
        array of shape [H x W x 3]

    scaling : int
        positive integer value > 0

    Returns
    -------

    rgbimg : numpy.ndarray
        three-dimensional array of shape [scaling*H x scaling*W x 3]
    '''


    #top left corner.
    rgbimg[0:scaling,0:scaling,:] = (rgbimg[0,scaling,:] + rgbimg[scaling,0,:] + rgbimg[scaling, scaling,:])/3.0
    #bottom right corner
    rgbimg[-scaling:,-scaling:,:] = (rgbimg[-1,-1-scaling, :] + rgbimg[-1-scaling, -1, :] + rgbimg[-1-scaling,-1-scaling,:])/3.0
    return rgbimg


def digit_to_rgb(X, scaling=3, shape = (), cmap = 'binary'):
    '''
    Takes as input an intensity array and produces a rgb image due to some color map

    Parameters
    ----------

    X : numpy.ndarray
        intensity matrix as array of shape [M x N]

    scaling : int
        optional. positive integer value > 0

    shape: tuple or list of its , length = 2
        optional. if not given, X is reshaped to be square.

    cmap : str
        name of color map of choice. default is 'binary'

    Returns
    -------

    image : numpy.ndarray
        three-dimensional array of shape [scaling*H x scaling*W x 3] , where H*W == M*N
    '''

    #create color map object from name string
    cmap = eval('matplotlib.cm.{}'.format(cmap))

    image = enlarge_image(vec2im(X,shape), scaling) #enlarge
    image = cmap(image.flatten())[...,0:3].reshape([image.shape[0],image.shape[1],3]) #colorize, reshape

    return image



def hm_to_rgb(R, X = None, scaling = 3, shape = (), sigma = 2, cmap = 'seismic', normalize = True, reduce_op='sum', reduce_axis=-1, model="VGG19"):

    '''
    Takes as input an intensity array and produces a rgb image for the represented heatmap.
    optionally draws the outline of another input on top of it.

    Parameters
    ----------

    R : numpy.ndarray
        the heatmap to be visualized, shaped [M x N]

    X : numpy.ndarray
        optional. some input, usually the data point for which the heatmap R is for, which shall serve
        as a template for a black outline to be drawn on top of the image
        shaped [M x N]

    scaling: int
        factor, on how to enlarge the heatmap (to control resolution and as a inverse way to control outline thickness)
        after reshaping it using shape.

    shape: tuple or list, length = 2
        optional. if not given, X is reshaped to be square.

    sigma : double
        optional. sigma-parameter for the canny algorithm used for edge detection. the found edges are drawn as outlines.

    cmap : str
        optional. color map of choice

    normalize : bool
        optional. whether to normalize the heatmap to [-1 1] prior to colorization or not.

    Returns
    -------

    rgbimg : numpy.ndarray
        three-dimensional array of shape [scaling*H x scaling*W x 3] , where H*W == M*N
    '''

    #create color map object from name string
    cmap = eval('cm.{}'.format(cmap))


    if reduce_op == 'sum':
        R = R.sum(axis=reduce_axis)
    elif reduce_op == 'absmax':
        pos_max = R.max(axis=reduce_axis)
        neg_max = (-R).max(axis=reduce_axis)
        abs_neg_max = -neg_max
        R = np.select([pos_max >= abs_neg_max, pos_max < abs_neg_max],
                        [pos_max, neg_max])
    else:
        raise NotImplementedError()
         
    
    if normalize:
        R = R / np.max(np.abs(R)) # normalize to [-1,1] wrt to max relevance magnitude
        R = (R + 1.)/2. # shift/normalize to [0,1] for color mapping

    R = enlarge_image(vec2im(R,shape), scaling)
    rgb = cmap(R.flatten())[...,0:3].reshape([R.shape[0],R.shape[1],3])
    #rgb = repaint_corner_pixels(rgb, scaling) #obsolete due to directly calling the color map with [0,1]-normalized inputs
    if model == 'CNN8':
        return rgb
    
    if not X is None: #compute the outline of the input

        if reduce_op == 'sum':
            X = X.sum(axis=reduce_axis)
        elif reduce_op == 'absmax':
            pos_max = X.max(axis=reduce_axis)
            neg_max = (-X).max(axis=reduce_axis)
            abs_neg_max = -neg_max
            X = np.select([pos_max >= abs_neg_max, pos_max < abs_neg_max],
                            [pos_max, neg_max])
        else:
            raise NotImplementedError()
        
        
        X = enlarge_image(vec2im(X,shape), scaling)
        xdims = X.shape
        Rdims = R.shape

        if not np.all(xdims == Rdims):
            print('transformed heatmap and data dimension mismatch. data dimensions differ?')
            print('R.shape = ',Rdims, 'X.shape = ', xdims)
            print('skipping drawing of outline\n')
        else:
            #edges = skimage.filters.canny(X, sigma=sigma)
            edges = skimage.feature.canny(X, sigma=sigma)
            edges = np.invert(np.dstack([edges]*3))*1.0
            rgb *= edges # set outline pixels to black color
    
    return rgb


def save_image(rgb_images, path, gap = 2):
    '''
    Takes as input a list of rgb images, places them next to each other with a gap and writes out the result.

    Parameters
    ----------

    rgb_images : list , tuple, collection. such stuff
        each item in the collection is expected to be an rgb image of dimensions [H x _ x 3]
        where the width is variable

    path : str
        the output path of the assembled image

    gap : int
        optional. sets the width of a black area of pixels realized as an image shaped [H x gap x 3] in between the input images

    Returns
    -------

    image : numpy.ndarray
        the assembled image as written out to path
    '''

    sz = []
    image = []
    #for i in xrange(len(rgb_images)): 
    for i in range(len(rgb_images)):
        if not sz:
            sz = rgb_images[i].shape
            image = rgb_images[i]
            gap = np.zeros((sz[0],gap,sz[2]))
            continue
        if not sz[0] == rgb_images[i].shape[0] and sz[1] == rgb_images[i].shape[2]:
            print('image',i, 'differs in size. unable to perform horizontal alignment')
            print('expected: Hx_xD = {0}x_x{1}'.format(sz[0],sz[1]))
            print('got     : Hx_xD = {0}x_x{1}'.format(rgb_images[i].shape[0],rgb_images[i].shape[1]))
            print('skipping image\n')
        else:
            image = np.hstack((image,gap,rgb_images[i]))

    image *= 255
    image = image.astype(np.uint8) 

    print('saving image to ', path)
    skimage.io.imsave(path,image)
    return image

########## 20181225 https://github.com/albermax/innvestigate/blob/master/innvestigate/utils/visualizations.py ###############

def project(X, output_range=(0, 1), absmax=None, input_is_postive_only=False):

#     X -= X.mean()
#     X = X / np.abs(X).max()
    if absmax is None:
        absmax = np.amax(np.abs(X))
    absmax = np.asarray(absmax)

    mask = absmax != 0
    if mask.sum() > 0:
        
        X[mask] /= absmax[mask, np.newaxis]

    if input_is_postive_only is False:
        X = (X+1)/2  # [0, 1]
    X = X.clip(0, 1)

    X = output_range[0] + (X * (output_range[1]-output_range[0]))
    return X





def graymap(X, **kwargs):
    return heatmap(X, cmap_type="gray", **kwargs)


def gamma(X, gamma = 0.5, minamp=0, maxamp=None):
    """
    apply gamma correction to an input array X
    while maintaining the relative order of entries,
    also for negative vs positive values in X.
    the fxn firstly determines the max
    amplitude in both positive and negative
    direction and then applies gamma scaling
    to the positive and negative values of the
    array separately, according to the common amplitude.
    :param gamma: the gamma parameter for gamma scaling
    :param minamp: the smallest absolute value to consider.
    if not given assumed to be zero (neutral value for relevance,
        min value for saliency, ...). values above and below
        minamp are treated separately.
    :param maxamp: the largest absolute value to consider relative
    to the neutral value minamp
    if not given determined from the given data.
    """

    #prepare return array
    Y = np.zeros_like(X)

    X = X - minamp # shift to given/assumed center
    if maxamp is None: maxamp = np.abs(X).max() #infer maxamp if not given
    X = X / maxamp # scale linearly

    #apply gamma correction for both positive and negative values.
    i_pos = X > 0
    i_neg = np.invert(i_pos)
    Y[i_pos] = X[i_pos]**gamma
    Y[i_neg] = -(-X[i_neg])**gamma

    #reconstruct original scale and center
    Y *= maxamp
    Y += minamp

    return Y


def clip_quantile(X, quantile=1):

    if not isinstance(quantile, (list, tuple)):
        quantile = (quantile, 100-quantile)

    low = np.percentile(X, quantile[0])
    high = np.percentile(X, quantile[1])
    X[X < low] = low
    X[X > high] = high

    return X


def batch_flatten(x):
    # Flattens all but the first dimensions of a numpy array, i.e. flatten each sample in a batch
    if not isinstance(x, np.ndarray):
        raise TypeError("Only applicable to Numpy arrays.")
    return x.reshape(x.shape[0], -1)


def preprocess(X, net):
    X = X.copy()
    X = net["preprocess_f"](X)
    return X


# def postprocess(X, color_conversion, channels_first):
#     X = X.copy()
#     X = iutils.postprocess_images(
#         X, color_coding=color_conversion, channels_first=channels_first)
#     return X


def image(X):
    X = X.copy()
    return project(X, absmax=255.0, input_is_postive_only=True)


def bk_proj(X):
    X = clip_quantile(X, 1)
    return project(X)


def heatmap(X,reduce_axis=-1, gamma_=0.9):
    X = gamma(X, minamp=0, gamma=gamma_)
    return heatmap_(X,reduce_axis=reduce_axis,reduce_op='sum')


def graymap(X):
    return graymap(np.abs(X), input_is_postive_only=True)

def heatmap_(X, cmap_type='seismic', reduce_op='sum', reduce_axis=-1, **kwargs):#'seismic'
    cmap = cm.get_cmap(cmap_type)
    #cmap = eval('matplotlib.cm.{}'.format(cmap_type))

    tmp = X
    shape = tmp.shape

    if reduce_op == 'sum':
        tmp = tmp.sum(axis=reduce_axis)
    elif reduce_op == 'absmax':
        pos_max = tmp.max(axis=reduce_axis)
        neg_max = (-tmp).max(axis=reduce_axis)
        abs_neg_max = -neg_max
        tmp = np.select([pos_max >= abs_neg_max, pos_max < abs_neg_max],
                        [pos_max, neg_max])
    else:
        raise NotImplementedError()

    ##20181227 lrp visualization like gradCAM : only_positive, threshold(tmp.mean)
    #tmp = tmp.clip(0)
    #condition = np.greater(tmp, tmp.mean(axis=reduce_axis))
    #tmp = condition*tmp
    tmp = project(tmp, output_range=(0, 255), input_is_postive_only=False, **kwargs).astype(np.int64)
    ##
    
    #tmp = project(tmp, output_range=(0, 255), **kwargs).astype(np.int64)

    tmp = cmap(tmp.flatten())[:, :3].T
    tmp = tmp.T

    shape = list(shape)
    shape[reduce_axis] = 3
    return tmp.reshape(shape).astype(np.float32)