#!/usr/bin/env python3.8
import os
import time
import numpy as np
import skimage, skimage.color, skimage.exposure, skimage.feature, skimage.filters, skimage.io, skimage.measure, skimage.transform
import matplotlib.pyplot as plt

################################################################################
# general image pre-procsesing functions
################################################################################


##########################
# general utility functions
##########################

def get_img(n=0, img_type='train') :
    """
    laod an image from the dataset
    WARNING : the images are BIG! fitting more than one or two into memory can
    be a challenge
    """
    img_name = '{}_{}.jpg'.format(img_type, str(n).zfill(2))
    img_path = os.path.join(os.getcwd(), 'data', img_type, img_name)
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

def show_img(img, title='') :
    """
    plots the image
    """
    fig = plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(title + ' ' + str(img.shape))
    plt.axis('off')
    plt.tight_layout()


##########################
# table registration
##########################

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
        if len(img.shape) == 3 :
            dwn = (dwn_f, dwn_f, 1)
        else :
            dwn = (dwn_f, dwn_f)
        img_smol = skimage.transform.downscale_local_mean(img, dwn)[:-1,:-1] #remove last pixel row, because downsampling produces artifacts on the edges
    else :
        dwn_f    = 1
        img_smol = img

    if len(img_smol.shape) > 2 :
        img_smol = skimage.color.rgb2gray(img_smol)
    else :
        img_smol = img

    corners = dwn_f * get_table_corners(img_smol)

    if not len(corners) :
        print("ERR: No corners detected!")
        return img, corners

    img_size = np.min(img.shape[0:2])
    dst  = np.array([[0,0], [img_size, 0], [img_size, img_size], [0, img_size]])

    tform = skimage.transform.estimate_transform('projective', corners, dst)
    img_tf = skimage.transform.warp(img, tform.inverse, output_shape=(img_size, img_size))

    return img_tf, corners

def apply_statistical_equalization(img, target_mean=0.7, target_std=12, range_type='squish') :
    """
    apply a mean and standard deviation-based normalization
    to ignore one of the statistics, pass None (otherwise a default value will
    be used)
    To prevent loss of information, the function squishes the new range using a
    sigmoid function centered around the desired mean.
    This behavior can be disabled by telling it to rescale by clipping : range_type='clip'
    """

    mm = np.mean(img, axis=(0,1)) # get current mean

    if target_std == None or target_std == 0:
        if target_mean == None:
            return img
        else :
            img_eq = img  + (target_mean - mm)
    else :
        ss = np.std(img, axis=(0,1)) # get current std
        if target_mean == None :
            img_eq = target_std/ss * (img - mm) + mm
        else :
            img_eq = target_std/ss * (img - mm) + target_mean


    if range_type == 'clip' :
        return (img_eq/img.max()).clip(0,1)
    elif target_mean == None :
        return 1/(1+np.exp(-4*img_eq/img.max() + 2))
    else:
        return 1/(1+np.exp(-4*img_eq/img.max() + 2*target_mean))


##########################
# player card analysis
##########################

def get_center_of_cards(img, sigma=1.5) :
    """
    extract the edges of the image -- assume the center of the cards corresponds to the centroid of the edges
    """
    img_edg      = skimage.feature.canny(skimage.color.rgb2gray(img), sigma=sigma)
    (X,Y)        = np.meshgrid(range(0, img.shape[1]), range(0, img.shape[0]))
    com          = np.array([Y[img_edg].sum(), X[img_edg].sum()])/np.count_nonzero(img_edg)

    return com

def check_card_back_presence(img, thresh=.4):
    """
    check for the presence of the pattern present on the backs of the cards
    using fast auto-correlation and a heuristic threshold
    """

    img_fft    = np.fft.fft2(skimage.filters.sobel(skimage.color.rgb2gray(img)))# fourier transform of the image

    auto_corr  = np.abs(np.fft.ifft2(img_fft * np.conjugate(img_fft)))          # fast auto-correlation
    auto_corr  = (auto_corr-auto_corr.min()) / np.ptp(auto_corr)                # normalize

    roi        = (slice(auto_corr.shape[0]//16, auto_corr.shape[0]//2),         # region of interest
                  slice(auto_corr.shape[1]//16, auto_corr.shape[1]//2))

    return np.any(auto_corr[roi] > thresh)


################################################################################
# utlilty classes
################################################################################

class PlayerSegment:
    def __init__(self, id=0, img=np.array([]), guess=None, is_rotated=False, has_folded=None):

        if guess :
            crop = slice(max(int(guess[0]-.125*img.shape[0]), 0), min(int(guess[0]+.125*img.shape[0]), img.shape[0])), \
                   slice(max(int(guess[1]-.125*img.shape[1]), 0), min(int(guess[1]+.125*img.shape[1]), img.shape[1]))
            com = get_center_of_cards(img[crop]) + np.array([crop[0].start, crop[1].start])
            crop = slice(max(int(com[0]-.13*img.shape[0]), 0), min(int(com[0]+.13*img.shape[0]), img.shape[0])), \
                   slice(max(int(com[1]-.13*img.shape[1]), 0), min(int(com[1]+.13*img.shape[1]), img.shape[1]))
            img = img[crop]

        if id and not is_rotated :
            if id == 1 :
                img = np.flip(img.transpose(1,0,2), 1)
            if id == 2 or id == 3 :
                img = np.flip(img, (0,1))
            if id == 4 :
                img = np.flip(img.transpose(1,0,2), 0)

        self.id         = id
        self.img        = img

        if has_folded is not None :
            self.has_folded = has_folded
        elif len(img):
            self.has_folded = check_card_back_presence(img)
        else :
            self.has_folded = False

    def show(self, ax=None):
        if not ax :
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        ax.imshow(self.img)
        ax.set_title("p{}: {}".format(self.id, "has folded" if self.has_folded else "is playing" ))
        ax.axis('off')

    def extract_player_cards(self, img, guess) :
        """
        takes a full table image and a guess at the center location of the player's cards
        returns an image segment centered around the player's cards
        """

        x   = slice(max(int(guess[1]-.125*img.shape[1]), 0), min(int(guess[1]+.125*img.shape[1]), img.shape[1]))
        y   = slice(max(int(guess[0]-.125*img.shape[0]), 0), min(int(guess[0]+.125*img.shape[0]), img.shape[0]))
        com = get_center_of_cards(img[y, x]) + np.array([y.start, x.start])
        x   = slice(max(int(com[1]-.13*img.shape[1]), 0), min(int(com[1]+.13*img.shape[1]), img.shape[1]))
        y   = slice(max(int(com[0]-.13*img.shape[0]), 0), min(int(com[0]+.13*img.shape[0]), img.shape[0]))
        return img[y, x]

class TableSegments:
    def __init__(self, img, is_registered=False, is_equalized=False):

        tic = time.time()

        if not is_registered :
            img, corners = register_table(img)
            print("---- %.3f seconds to register table ---" % (time.time() - tic))
            assert len(corners), "table corners could not be detected"
            tic = time.time()

        if not is_equalized :
            img = apply_statistical_equalization(img, target_std=None)
            print("---- %.3f seconds to equalize table ---" % (time.time() - tic))
            tic = time.time()


        # player 1
        guess = (1000/2000*img.shape[0], 1750/2000*img.shape[1])
        self.player = [PlayerSegment(1, img, guess)]

        # player 2
        guess = ( 250/2000*img.shape[0], 1425/2000*img.shape[1])
        self.player.append(PlayerSegment(2, img, guess))

        # player 3
        guess = ( 250/2000*img.shape[0],  550/2000*img.shape[1])
        self.player.append(PlayerSegment(3, img, guess))

        # player 4
        guess = (1050/2000*img.shape[0],  250/2000*img.shape[1])
        self.player.append(PlayerSegment(4, img, guess))

        # table
        xl, xh = int( 200/2000*img.shape[1]), int(1800/2000*img.shape[1])
        yl, yh = int(1500/2000*img.shape[0]), int(2000/2000*img.shape[0])
        self.T = img[yl:yh, xl:xh]

        # chips
        xl, xh = int( 500/2000*img.shape[1]), int(1500/2000*img.shape[1])
        yl, yh = int( 500/2000*img.shape[0]), int(1500/2000*img.shape[0])
        self.c = img[yl:yh, xl:xh]
        print("---- %.3f seconds to segment table ---" % (time.time() - tic))

    def show(self, title='') :
        fig = plt.figure()
        plt.suptitle(title)

        ax1 = fig.add_subplot(3,4,(1,4))
        ax1.imshow(self.c)
        ax1.set_title("Chips")
        ax1.axis('off')

        ax2 = fig.add_subplot(3,4,(5,8))
        ax2.imshow(self.T)
        ax2.set_title("T")
        ax2.axis('off')

        for i in range(4) :
            ax = fig.add_subplot(3,4,9+i)
            self.player[i].show(ax)

        fig.tight_layout()


################################################################################
# testing
################################################################################

if __name__ == '__main__' :
    training_set = list(range(28)) + [99]

    for n in [1,2,3,8,21,22] :
    # for n in training_set:

        print("img", n)
        # get img
        tic = time.time()
        img =  get_img(n)[::2,::2,:]
        print("--- %.3f seconds to load image ---" % (time.time() - tic))

        tic = time.time()
        segments = TableSegments(img)
        print("--- %.3f seconds to segment image ---" % (time.time() - tic))


        segments.show("train_{}.jpg".format(str(n).zfill(2)))

    plt.show()
