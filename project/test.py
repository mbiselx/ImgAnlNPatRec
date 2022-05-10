#!/usr/bin/env python3.8
import os
import numpy as np
import skimage, skimage.color, skimage.exposure, skimage.feature, skimage.filters, skimage.io, skimage.measure, skimage.transform
import matplotlib.pyplot as plt

training_set = list(range(28)) + [99]

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

def make_smol(img, dwn_f=10, gray=True, rotate=False) :
    """
    downsample and grayscale a given image to reduce the computation time
    can also apply a (random) rotation, if desired (for testing/training)
    """
    if len(img.shape) == 3 :
        dwn = (dwn_f, dwn_f, 1)
    else :
        dwn = (dwn_f, dwn_f)


    img_smol = skimage.transform.downscale_local_mean(img, dwn)
    # img_smol = skimage.transform.rescale(img, 1/dwn_f, anti_aliasing=True, multichannel=mc) #channel_axis=2) #this should work, but it doesn't
    # img_smol = skimage.transform.resize(img, (img.shape[0] // dwn_f, img.shape[1] // dwn_f, img.shape[2]), anti_aliasing=True) #this should work too, but it doesn't
    # img_smol = img[::dwn_f, ::dwn_f, :] # zero-order is really bad but seems to work-ish

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

def check_card_back_presence(img, thresh=.4):
    """
    check for the presence of the pattern present on the backs of the cards
    usin autocorrelation and a heuristic threshold
    """
    # img_fft    = np.fft.fft2(skimage.color.rgb2gray(img))
    img_fft    = np.fft.fft2(skimage.filters.sobel(skimage.color.rgb2gray(img)))

    auto_corr  = np.abs(np.fft.ifft2(img_fft * np.conjugate(img_fft)))
    auto_corr  = (auto_corr-auto_corr.min()) / np.ptp(auto_corr)                # normalize
    roi        = (slice(auto_corr.shape[0]//16, auto_corr.shape[0]//2),         # region of interest
                  slice(auto_corr.shape[1]//16, auto_corr.shape[1]//2))

    # show(auto_corr[roi])

    return np.any(auto_corr[roi] > thresh)                                      # get pattern presence (i.e. autocorr. peaks)

class PlayerSegment:
    def __init__(self, id=0, img=np.array([]), has_folded=None):
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

class TableSegments:
    def __init__(self, p=[PlayerSegment()], T=[], c=[]):
        self.player = p
        self.T = T
        self.c = c

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

def get_center_of_mass(img):
    """
    calculate center of mass of a grayscale picture
    """
    img_gray = skimage.color.rgb2gray(img)
    (X,Y) = np.meshgrid(range(0, img.shape[1]), range(0, img.shape[0]))
    com   = np.array([(Y*img_gray).sum(), (X*img_gray).sum()])/img_gray.sum()
    return com

    # img_hsv      = skimage.color.rgb2hsv(img)
    # (X,Y)        = np.meshgrid(range(0, img.shape[1]), range(0, img.shape[0]))
    #
    # black_idx    = img_hsv[:,:,2] < .25
    # red_idx      = np.logical_and(np.logical_or(img_hsv[:,:,0] < .05, img_hsv[:,:,0] > .95), img_hsv[:,:,1] > .6)
    # relevant_idx = np.logical_or(red_idx, black_idx)
    # x, y         = X[relevant_idx].sum(), Y[relevant_idx].sum()
    #
    # return np.array([y,x])/np.count_nonzero(relevant_idx)

def get_center_of_cards(img, thresh=.3) :
    """
    extract the edges of the image -- assume the center of the cards corresponds to the centroid of the edges
    """
    # img_sobel    = skimage.filters.sobel(skimage.color.rgb2gray(img))
    # img_edg      = img_sobel > thresh*img_sobel.max()
    img_edg      = skimage.feature.canny(skimage.color.rgb2gray(img), sigma=3)
    (X,Y)        = np.meshgrid(range(0, img.shape[1]), range(0, img.shape[0]))
    com          = np.array([Y[img_edg].sum(), X[img_edg].sum()])/np.count_nonzero(img_edg)

    # idx = img_edg.nonzero()
    # yl, yr = np.min(idx[0]), np.max(idx[0])
    # xl, xr = np.min(idx[1]), np.max(idx[1])
    # y = (yr-yl)/2 + yl
    # x = (xr-xl)/2 + xl
    # com = np.array([y,x])

    return com

def get_player_img(img, guess):
    """
    takes a full table image and a guess at the location of the player's cards
    returns an image segment centered around the player's cards
    (unless the cards are face downwards -- then it returns nonsense)
    """
    x   = slice(max(int(guess[1]-.125*img.shape[1]), 0), min(int(guess[1]+.125*img.shape[1]), img.shape[1]))
    y   = slice(max(int(guess[0]-.125*img.shape[0]), 0), min(int(guess[0]+.125*img.shape[0]), img.shape[0]))
    com = get_center_of_cards(img[y, x]) + np.array([y.start, x.start])
    x   = slice(max(int(com[1]-.13*img.shape[1]), 0), min(int(com[1]+.13*img.shape[1]), img.shape[1]))
    y   = slice(max(int(com[0]-.13*img.shape[0]), 0), min(int(com[0]+.13*img.shape[0]), img.shape[0]))
    return img[y, x]

def segment_table_simple(img) :
    """
    segments the table into [p1 p2 p3 p4 t chips]
    returns a TableSegments
    """
    output = TableSegments()

    # player 1
    guess = slice(int( 750/2000*img.shape[0]), int(1250/2000*img.shape[1])), \
            slice(int(1500/2000*img.shape[1]), int(2000/2000*img.shape[1]))
    p1     = np.flip(img[guess].transpose(1,0,2), 1)
    output.player = [PlayerSegment(id=1, img=p1)]

    # player 2
    guess = slice(int(   0/2000*img.shape[0]), int( 500/2000*img.shape[1])), \
            slice(int(1150/2000*img.shape[1]), int(1650/2000*img.shape[1]))
    p2     = np.flip(img[guess], (0,1))
    output.player.append(PlayerSegment(id=2, img=p2))

    # player 3
    guess = slice(int(   0/2000*img.shape[0]), int( 500/2000*img.shape[1])), \
            slice(int( 300/2000*img.shape[1]), int( 800/2000*img.shape[1]))
    p3     = np.flip(img[guess], (0,1))
    output.player.append(PlayerSegment(id=3, img=p3))

    # player 4
    guess = slice(int( 800/2000*img.shape[0]), int(1300/2000*img.shape[1])), \
            slice(int(   0/2000*img.shape[1]), int( 500/2000*img.shape[1]))
    p4     = np.flip(img[guess].transpose(1,0,2), 0)
    output.player.append(PlayerSegment(id=4, img=p4))

    # table
    xl, xh = int( 200/2000*img.shape[1]), int(1800/2000*img.shape[1])
    yl, yh = int(1500/2000*img.shape[0]), int(2000/2000*img.shape[0])
    output.T = img[yl:yh, xl:xh]

    # chips
    xl, xh = int( 500/2000*img.shape[1]), int(1500/2000*img.shape[1])
    yl, yh = int( 500/2000*img.shape[0]), int(1500/2000*img.shape[0])
    output.c = img[yl:yh, xl:xh]

    return output

def segment_table(img) :
    """
    segments the table into [p1 p2 p3 p4 t chips]
    returns a TableSegments
    """
    output = TableSegments()

    # player 1
    guess = (1000/2000*img.shape[0], 1750/2000*img.shape[1])
    p1     = np.flip(get_player_img(img, guess).transpose(1,0,2), 1)
    output.player = [PlayerSegment(id=1, img=p1)]

    # player 2
    guess = ( 250/2000*img.shape[0], 1425/2000*img.shape[1])
    p2     = np.flip(get_player_img(img, guess), (0,1))
    output.player.append(PlayerSegment(id=2, img=p2))

    # player 3
    guess = ( 250/2000*img.shape[0],  550/2000*img.shape[1])
    p3     = np.flip(get_player_img(img, guess), (0,1))
    output.player.append(PlayerSegment(id=3, img=p3))

    # player 4
    guess = (1050/2000*img.shape[0],  250/2000*img.shape[1])
    p4     = np.flip(get_player_img(img, guess).transpose(1,0,2), 0)
    output.player.append(PlayerSegment(id=4, img=p4))

    # table
    xl, xh = int( 200/2000*img.shape[1]), int(1800/2000*img.shape[1])
    yl, yh = int(1500/2000*img.shape[0]), int(2000/2000*img.shape[0])
    output.T = img[yl:yh, xl:xh]

    # chips
    xl, xh = int( 500/2000*img.shape[1]), int(1500/2000*img.shape[1])
    yl, yh = int( 500/2000*img.shape[0]), int(1500/2000*img.shape[0])
    output.c = img[yl:yh, xl:xh]

    return output

def get_mean_range(img, thresh_lo=None, thresh_hi=None):
    img_gray   = skimage.color.rgb2gray(img)

    if thresh_lo == None :
        thresh_lo = np.percentile(img_gray, 5)
    if thresh_hi == None :
        thresh_hi = np.percentile(img_gray, 90)

    black_idx  = np.logical_and(img_gray < thresh_lo, np.all(img < .4*img.max(), axis=2))
    mean_black = img[black_idx].mean(0) if np.any(black_idx) else np.array([])

    white_idx  = img_gray > thresh_hi
    mean_white = img[white_idx].mean(0)

    return mean_black, mean_white

def apply_linear_rescale(img, mean_black=np.array([0,0,0]), mean_white=np.array([1,1,1])) :
    return ((img - mean_black) / (mean_white-mean_black)).clip(0,1)

def apply_logistic_rescale(img, mean_black=np.array([0,0,0]), mean_white=np.array([1,1,1])) :
    if not len(mean_black) :
        mean_black = np.array([0,0,0])
    if not len(mean_white) :
        mean_white = np.array([1,1,1])

    x  = (img - mean_black) / (mean_white-mean_black) # linear rescale
    return 1/(1+np.exp(-4*x + 2)) # sigmoid suqishing

def equalize_img(img) :
    """
    equalizes the color of the table
    """
    # img_ref = get_img(img_type='out', n=0)
    # img = skimage.exposure.match_histograms(img, img_ref, multichannel=True)

    # xl, xh = int( 300/2000*img.shape[0]), int(1000/2000*img.shape[0])
    # yl, yh = int(1500/2000*img.shape[0]), int(1900/2000*img.shape[0])
    #
    # img_patch = skimage.filters.gaussian(img[yl:yh, xl:xh], sigma=10, multichannel=True)
    # img = ((skimage.img_as_float(img)-img_patch.min(axis=(0, 1))) / (img_patch.max(axis=(0, 1)) - img_patch.min(axis=(0, 1)))).clip(0, 1)
    # img[yl:yh, xl:xh] = img_patch/skimage.dtype_limits(img_patch)[1] # check

    mean_black, mean_white = get_mean_range(img)
    return apply_linear_rescale(img, mean_black, mean_white)

def equalize_table(segments=TableSegments()) :
    """
    equalize the color of the table segments
    """
    mean_black, mean_white = [], []
    for p in segments.player :
        if not p.has_folded :
            mb, mw = get_mean_range(p.img)
            if len(mb) : mean_black.append(mb)
            if len(mw) : mean_white.append(mw)
    mb, mw = get_mean_range(segments.T)
    if len(mb) : mean_black.append(mb)
    if len(mw) : mean_white.append(mw)

    segments.mean_black = np.array(mean_black).mean(0)
    segments.mean_white = np.array(mean_white).mean(0)

    for p in segments.player :
        p.img = apply_logistic_rescale(p.img, segments.mean_black, segments.mean_white)
    segments.T = apply_logistic_rescale(segments.T , segments.mean_black, segments.mean_white)
    segments.c = apply_logistic_rescale(segments.c , segments.mean_black, segments.mean_white)

    return segments

##white point is at (70, 66)
# img_ref = get_img(img_type='out', n=1)
# com = get_center_of_mass(img_ref)
# print(com)
# show(img_ref)
# plt.scatter(com[0], com[1])
# plt.tight_layout()
# plt.show()
# exit()


# a = np.array([[  0,  0,  0,  0,  0,  0,  0],
#               [  0,  0, -1,  0,  3,  0,  0],
#               [  0,  0,  1,  0,  1,  0,  0],
#               [  0,  0,  0,  1,  0,  0,  0],
#               [  0,  0,  0,  0,  0,  0,  0]])
# print(a)
# b = a.nonzero()
# y = slice(np.min(b[0]), np.max(b[0])+1)
# x = slice(np.min(b[1]), np.max(b[1])+1)
# print(a[y,x])
# exit()





if __name__ == '__main__' :
    import time

    m = 0;

    for n in training_set:
    # for n in [99] :
    # for n in [1,2,3,8,21,22] :
    # for n in [1,2,3,8,11,15,22] :

        print("img", n)

        # # get img
        # tic = time.time()
        # test =  get_img(n)[::2,::2,:]
        # print("--- %.3f seconds to load image ---" % (time.time() - tic))
        #
        # # register & get corners
        # tic = time.time()
        # table, corners = register_table(test)
        # print("--- %.3f seconds to register table ---" % (time.time() - tic))
        # if not len(corners) :
        #     print("skipping image")
        #     continue # assume we were not able to detect the table
        #
        # # save to output
        # save_img(table, n, img_type="table")

        # load in the registered table
        tic = time.time()
        table = get_img(n, img_type='table')
        print("--- %.3f seconds to load table ---" % (time.time() - tic))

        # segment the teable
        tic = time.time()
        # segments = segment_table_simple(table)
        segments = segment_table(table)
        print("--- %.3f seconds to segment table ---" % (time.time() - tic))

        # get color equalization parameters
        tic = time.time()
        segments = equalize_table(segments)
        print("--- %.3f seconds to recolor table ---" % (time.time() - tic))

        # # equalizes the color of the tables
        # tic = time.time()
        # table_eq = apply_logistic_rescale(table , segments.mean_black, segments.mean_white)
        # print("--- %.3f seconds to equalize color ---" % (time.time() - tic))
        #
        # # save to output
        # save_img(table_eq, n, img_type="table_eq")

        # show image
        # segments.show("train_{}.jpg".format(str(n).zfill(2)))

        # save to output
        for i in range(4) :
            if not segments.player[i].has_folded :
                save_img(segments.player[i].img, m, img_type="cards")
            else :
                save_img(segments.player[i].img, m, img_type="folds")
            m = m+1

        # clear memory
        # del test
        # del table
        # del segments

    plt.tight_layout()
    plt.show()
