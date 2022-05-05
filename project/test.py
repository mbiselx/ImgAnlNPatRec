#!/usr/bin/env python3.8

import os
import numpy as np
import skimage, skimage.io, skimage.color, skimage.filters, skimage.transform, skimage.exposure
import matplotlib.pyplot as plt


def get_img(n=0, img_type='train') :
    """
    laod an image from the dataset
    WARNING : the images are BIG! fitting more than one or two into memory can
    be a challenge
    """
    img_name = '{}_{}.jpg'.format(img_type, str(n).zfill(2))
    img_path = os.path.join(os.getcwd(), 'data', img_type, img_name)
    # print("Getting image :", img_path)
    img = skimage.io.imread(img_path)
    return img

def save_img(img, n=0, img_type='out') :
    """
    laod an image from the dataset
    """
    img_name = '{}_{}.jpg'.format(img_type, str(n).zfill(2))
    img_path = os.path.join(os.getcwd(), 'data', img_type)

    if not os.path.exists(img_path) :
        os.makedirs(img_path)

    skimage.io.imsave(os.path.join(img_path, img_name), skimage.img_as_ubyte(img))

def make_smol(img, dwn_f=10., gray=True, rotate=False) :
    """
    downsample and grayscale a given image to reduce the computation time
    can also apply a (random) rotation, if desired (for testing/training)
    """
    # if len(img.shape) == 3 :
    #     mc = True
    # else :
    #     mc = False

    # print(1/dwn_f)
    img_smol = skimage.transform.downscale_local_mean(img, (dwn_f, dwn_f, 1))
    # img_smol = skimage.transform.downscale_local_mean(img, (dwn_f, dwn_f, 1)).astype(np.uint8)#[:-1, :-1]
    # img_smol = skimage.transform.rescale(img, 1/dwn_f, anti_aliasing=True, multichannel=mc) #channel_axis=2) #this should work, but it doesn't
    # img_smol = skimage.transform.resize(img, (img.shape[0] // dwn_f, img.shape[1] // dwn_f, img.shape[2]), anti_aliasing=True) #this should work too, but it doesn't
    # img_smol = img[::dwn_f, ::dwn_f, :] # zero-order is really bad but seems to work-ish

    # img_smol = img.reshape()
    # show(img_smol)
    # plt.show()

    if gray and len(img.shape) == 3 :
        img_smol = skimage.color.rgb2gray(img_smol)

    if rotate :
        if type(rotate) is bool :
            rotate = 4 - 8*np.random.random()
        img_smol = skimage.transform.rotate(img_smol, rotate, mode='reflect')
        return img_smol, rotate
    return img_smol

def show(img, title='') :
    """
    plots the image
    """
    fig = plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(title + ' ' + str(img.shape))
    plt.axis('off')

def get_intersect(d1,a1, d2,a2) :
    """
    gets the intersection of two lines described by their hough transform
    d: distance
    a: angle
    returns an empty list if no intersection
    """
    try :
        p1 = d1 * np.array([np.cos(a1), np.sin(a1)])
        p2 = d2 * np.array([np.cos(a2), np.sin(a2)])
        A = np.array([[-np.sin(a1), -np.sin(a2)], [np.cos(a1), np.cos(a2)]])
        c = np.linalg.solve(A, (p1-p2))
        out = (p1 - c[0]*A[:,0] + p2 + c[1]*A[:,1])/2
        # print(out, p1+p2)
        return out
        # return p1+p2 # why does this not work????
    except Exception as e:
        return []

def sort_quadrants(pts):
    """
    sorts a list of exactly four points into their respective quadrants
    """

    if not (pts.shape == (4, 2)) :
        return []

    out = np.zeros((4, 2))
    mid = np.mean(pts, axis=0)
    # find upper left
    out[0,:] = pts[np.logical_and(pts[:,0] < mid[0], pts[:,1] < mid[1])]
    # find upper right
    out[1,:] = pts[np.logical_and(pts[:,0] > mid[0], pts[:,1] < mid[1])]
    # find lower right
    out[2,:] = pts[np.logical_and(pts[:,0] > mid[0], pts[:,1] > mid[1])]
    # find lower left
    out[3,:] = pts[np.logical_and(pts[:,0] < mid[0], pts[:,1] > mid[1])]

    return out

def get_table_corners(img) :
    """
    returns the detected corners of the table
    """
    # get edges of img
    img_sobel = skimage.filters.sobel(img)
    img_edg   = (img_sobel > skimage.filters.threshold_otsu(img_sobel)).astype(np.uint8)

    # find straight lines
    _ , angles, dists = skimage.transform.hough_line_peaks(*skimage.transform.hough_line(img_edg), num_peaks=4)

    # get all intersections
    intersects = []
    for i in range(len(angles)-1) :
        [intersects.append(get_intersect(dists[i],angles[i], d,a)) for d, a in zip(dists[i+1:], angles[i+1:])]
    intersects = np.array(list(filter(np.any, intersects)))

    # only take intersections in image area
    intersects = intersects[np.logical_and(np.logical_and(intersects[:,0] > 0, intersects[:,0] < img.shape[1]),
                                           np.logical_and(intersects[:,1] > 0, intersects[:,1] < img.shape[0]))]

    # sort the intersections into their respective quadrants
    corners = sort_quadrants(intersects)

    return corners

def register_table(img) :
    """
    performs registration of the table based on the position of the corners
    """
    dwn_f = int(np.ceil(np.max(img.shape)/300)) # the maximum dimension of the image to process should not be over 300 pixels
    if dwn_f > 1 :
        img_smol = make_smol(img, dwn_f)[:-1,:-1] #remove last pixel row, because downsampling produces artifacts on the edges
    else :
        dwn_f    = 1
        if len(img.shape) > 2 :
            img_smol = skimage.color.rgb2gray(img)
        else :
            img_smol = img

    corners = dwn_f * get_table_corners(img_smol)

    if not len(corners) :
        print("ERR: No corners detected!")
        return img, corners

    img_size = np.min(img.shape[0:2])
    dst  = np.array([[0,0], [img_size, 0], [img_size, img_size], [0, img_size]])

    tform = skimage.transform.estimate_transform('projective', corners, dst)
    img_tf = skimage.transform.warp(img, tform.inverse)[0:img_size, 0:img_size]

    return img_tf, corners

class TableSegments:
    def __init__(self, p=[], T=[], c=[]):
        self.p = p
        self.T = T
        self.c = c


def segment_table(img) :
    """
    segments the table :
    p1 p2 p3 p4 t chips
    """

    # player 1
    xl, xh = int(1500/2000*img.shape[1]), int(2000/2000*img.shape[1])
    yl, yh = int( 700/2000*img.shape[0]), int(1200/2000*img.shape[0])
    p1     = np.flip(img[yl:yh, xl:xh].transpose(1,0,2), 1)

    # player 2
    xl, xh = int(1200/2000*img.shape[1]), int(1700/2000*img.shape[1])
    yl, yh = int(   0/2000*img.shape[0]), int( 500/2000*img.shape[0])
    p2     = np.flip(img[yl:yh, xl:xh], (0,1))

    # player 3
    xl, xh = int( 300/2000*img.shape[1]), int( 800/2000*img.shape[1])
    yl, yh = int(   0/2000*img.shape[0]), int( 500/2000*img.shape[0])
    p3     = np.flip(img[yl:yh, xl:xh], (0,1))

    # player 4
    xl, xh = int(   0/2000*img.shape[1]), int( 500/2000*img.shape[1])
    yl, yh = int( 800/2000*img.shape[0]), int(1300/2000*img.shape[0])
    p4     = np.flip(img[yl:yh, xl:xh].transpose(1,0,2), 0)

    # table
    xl, xh = int( 200/2000*img.shape[1]), int(1800/2000*img.shape[1])
    yl, yh = int(1500/2000*img.shape[0]), int(2000/2000*img.shape[0])
    T      = img[yl:yh, xl:xh]

    # chips
    xl, xh = int( 500/2000*img.shape[1]), int(1500/2000*img.shape[1])
    yl, yh = int( 500/2000*img.shape[0]), int(1500/2000*img.shape[0])
    c      = img[yl:yh, xl:xh]

    output = TableSegments([p1, p2, p3, p4], T, c)

    return output



def equalize_table(img) :
    """
    equalizes the color of the table
    """
    # img_ref = get_img(img_type='out', n=0)
    # img = skimage.exposure.match_histograms(img, img_ref, multichannel=True)

    xl, xh = int( 300/2000*img.shape[0]), int(1000/2000*img.shape[0])
    yl, yh = int(1500/2000*img.shape[0]), int(1900/2000*img.shape[0])

    img_patch = skimage.filters.gaussian(img[yl:yh, xl:xh], sigma=10, multichannel=True)
    img = ((skimage.img_as_float(img)-img_patch.min(axis=(0, 1))) / (img_patch.max(axis=(0, 1)) - img_patch.min(axis=(0, 1)))).clip(0, 1)
    # img[yl:yh, xl:xh] = img_patch/skimage.dtype_limits(img_patch)[1] # check

    return img


# ###white point is (400, 1700)
# img_ref = get_img(img_type='out', n=0)
# show(img_ref)
# plt.tight_layout()
# plt.show()
# exit()

if __name__ == '__main__' :
    import time

    n = 0
    for n in [2,3] :#[1,2,3,8,11,22, 98, 99] :
    # for n in [1,2,3,8,11,22, 98, 99] :

        print("img", n)

        #get img
        tic = time.time()
        test =  get_img(n)[::2,::2,:]
        print("--- %.3f seconds to load image ---" % (time.time() - tic))

        # register & get corners
        tic = time.time()
        table, corners = register_table(test)
        print("--- %.3f seconds to register table ---" % (time.time() - tic))
        if not len(corners) :
            print("skipping image")
            continue # assume we were not able to detect the table

        tic = time.time()
        segments = segment_table(table)
        print("--- %.3f seconds to segment table ---" % (time.time() - tic))


        # show(segments[4])
        # fig, axes = plt.subplots(1,4)
        # [(ax.imshow(p), ax.axis('off')) for ax, p in zip(axes, segments[0:4])]


        fig = plt.figure()

        ax1 = fig.add_subplot(3,4,(1,4))
        ax1.imshow(segments.c)
        ax1.set_title("Chips")
        ax1.axis('off')

        ax1 = fig.add_subplot(3,4,(5,8))
        ax1.imshow(segments.T)
        ax1.set_title("T")
        ax1.axis('off')

        for i in range(4) :
            ax = fig.add_subplot(3,4,9+i)
            ax.imshow(segments.p[i])
            ax.set_title("p{}".format(i+1))
            ax.axis('off')


        # equaze image color
        # tic = time.time()
        # table = equalize_table(table)
        # print("--- %.3f seconds to equalize img table ---" % (time.time() - tic))

        # save to output
        # save_img(table, n)

        # clear memory
        del test, table

    # show(test)
    # plt.scatter(corners[:,0], corners[:,1], color='r')
    # show(table)
    #
    plt.tight_layout()
    plt.show()
