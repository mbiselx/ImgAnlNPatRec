#!/usr/bin/env python3.8
import os
import time
import numpy as np
import skimage
from skimage import color, exposure, feature, filters, io, measure, morphology, transform
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

# homemate stuff
import evaluate_cards

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
    img = io.imread(img_path)
    return img

def save_img(img, n=0, img_type='out') :
    """
    laod an image from the dataset
    """
    img_name = '{}_{}.jpg'.format(img_type, str(n).zfill(2))
    img_path = os.path.join(os.getcwd(), 'data', img_type)

    if not os.path.exists(img_path) :
        os.makedirs(img_path)

    io.imsave(os.path.join(img_path, img_name), skimage.img_as_ubyte(img))

def show_img(img, title='', ax=None) :
    """
    plots the image
    """
    if not ax :
        fig = plt.figure()
        plt.imshow(img, cmap='gray')
        plt.title(title + ' ' + str(img.shape))
        plt.axis('off')
    else :
        ax.imshow(img, cmap='gray')
        ax.set_title(title + ' ' + str(img.shape))
        ax.axis('off')
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
    img_sobel = filters.sobel(img)
    img_edg   = (img_sobel > filters.threshold_otsu(img_sobel)).astype(np.uint8)

    # find straight lines
    _ , angles, dists = transform.hough_line_peaks(*transform.hough_line(img_edg), num_peaks=4)

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
        img_smol = transform.downscale_local_mean(img, dwn)[:-1,:-1] #remove last pixel row, because downsampling produces artifacts on the edges
    else :
        dwn_f    = 1
        img_smol = img

    if len(img_smol.shape) > 2 :
        img_smol = color.rgb2gray(img_smol)
    else :
        img_smol = img

    corners = dwn_f * get_table_corners(img_smol)

    if not len(corners) :
        print("ERR: No corners detected!")
        return img, corners

    img_size = np.min(img.shape[0:2])
    dst  = np.array([[0,0], [img_size, 0], [img_size, img_size], [0, img_size]])

    tform = transform.estimate_transform('projective', corners, dst)
    img_tf = transform.warp(img, tform.inverse, output_shape=(img_size, img_size))

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
    img_edg      = feature.canny(color.rgb2gray(img), sigma=sigma)
    (X,Y)        = np.meshgrid(range(0, img.shape[1]), range(0, img.shape[0]))
    com          = np.array([Y[img_edg].sum(), X[img_edg].sum()])/np.count_nonzero(img_edg)

    return com

def check_card_back_presence(img, thresh=.4):
    """
    check for the presence of the pattern present on the backs of the cards
    using fast auto-correlation and a heuristic threshold
    """

    img_fft    = np.fft.fft2(filters.sobel(color.rgb2gray(img)))# fourier transform of the image

    auto_corr  = np.abs(np.fft.ifft2(img_fft * np.conjugate(img_fft)))          # fast auto-correlation
    auto_corr  = (auto_corr-auto_corr.min()) / np.ptp(auto_corr)                # normalize

    roi        = (slice(auto_corr.shape[0]//16, auto_corr.shape[0]//2),         # region of interest
                  slice(auto_corr.shape[1]//16, auto_corr.shape[1]//2))

    return np.any(auto_corr[roi] > thresh)


##########################
# chips analysis
##########################

def clean_binary_img(img):
    """
    use morphology to get rid of small garbage on binary images
    """
    footprint_closing = morphology.disk(20/2000*img.shape[0])
    footprint_opening = morphology.disk(22/2000*img.shape[0])
    return morphology.binary_opening(morphology.binary_closing(img,footprint_closing ),footprint_opening)

def get_hsv_channels(img):

    hsv_img = color.rgb2hsv(img)
    hue_img = hsv_img[:,:,0] # Hue
    sat_img = hsv_img[:,:,1] # Saturation
    value_img = hsv_img[:,:,2] # Value

    return hue_img,sat_img,value_img

def count_chips(img, show=False):
    # show_img(img, " input")
    # find the coordinates of local maxima which
    distance = ndi.distance_transform_edt(img)
    if show :
        show_img(distance, "distance_transform")
        plt.show()
    # each coordinate correspond to a chip
    coords = feature.peak_local_max(distance, min_distance=int(20/1000*img.shape[0]))

    return len(coords)

################################################################################
# utlilty classes
################################################################################

class PlayerSegment:
    def __init__(self, id=0, img=np.array([]), guess=None, is_rotated=False, has_folded=None):
        """
        initializes the 'player' table segment
        """

        # preprocessing
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
        self.has_folded = has_folded
        self.cards      = []
        self.cards_loc  = []

    def get_cards(self, card_space=None) :
        """
        processes the image segement to detect the value of the cards.
        returns the detected cards as a list
        [card1 card2]
        """

        # check to see if there even are any cards to detect
        if self.has_folded is None :
            self.has_folded = check_card_back_presence(self.img)

        if self.has_folded :
            return []

        # use the 'evaluate_cards' module to evaluate the cards
        if not card_space :
            card_space = evaluate_cards.load_card_space("cardspace_20.pkl")
        detected_cards = evaluate_cards.guess_cards(self.img, card_space, nb_cards=2)

        #extract the relevant information from the modeule's output
        self.cards, self.cards_loc = zip(*[(c[0], c[1]) for c in detected_cards])
        return self.cards

    def show(self, ax=None):
        if not ax :
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        ax.imshow(self.img)
        for card, loc in zip(self.cards, self.cards_loc) :
            ax.text(loc[1], loc[0], card, color='dodgerblue', size=15)
        ax.set_title("p{}: {}".format(self.id, "has folded" if self.has_folded else ("is playing")))
        ax.axis('off')

    def save(self, n=0):
        save_img(self.img, n, img_type=('folds' if self.has_folded else 'cards')+str(self.id))

class CommunitySegment:
    def __init__(self, img, cropped=False):
        """
        initializes the 'community cards' table segment
        """
        # preprocessing
        if not cropped :
            crop = slice(int(1500/2000*img.shape[0]), int(2000/2000*img.shape[0])), \
                   slice(int( 200/2000*img.shape[1]), int(1800/2000*img.shape[1]))
            img = img[crop]
        self.img = img

    def get_cards(self, card_space=None) :
        """
        processes the image segement to detect the value of the cards.
        returns the detected cards as a list
        [card1 card2]
        """

        # use the 'evaluate_cards' module to evaluate the cards
        if not card_space :
            card_space = evaluate_cards.load_card_space("cardspace_20.pkl")
        detected_cards = evaluate_cards.guess_cards(self.img, card_space, nb_cards=5)

        #extract the relevant information from the modeule's output
        self.cards, self.cards_loc = zip(*[(c[0], c[1]) for c in detected_cards])
        return self.cards

    def show(self, ax=None):
        if not ax :
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        ax.imshow(self.img)
        for card, loc in zip(self.cards, self.cards_loc) :
            ax.text(loc[1], loc[0], card, color='dodgerblue', size=15)
        ax.set_title("community cards")
        ax.axis('off')

    def save(self, n=0):
        save_img(self.img, n, img_type='cardsc')


class ChipsSegment:
    def __init__(self, img, cropped=False):
        """
        initializes the 'chips' table segment
        """

        # preprocessing
        if not cropped :
            crop = slice(int( 500/2000*img.shape[0]), int(1500/2000*img.shape[0])), \
                   slice(int( 500/2000*img.shape[1]), int(1500/2000*img.shape[1]))
            img = img[crop]
        self.img = img

        self.nb_chips = []
        self.centers  = []
        self.radii    = 70/1000*self.img.shape[0]

    def get_nb_chips(self):
        """
        processes the image segement to detect the presence of chips.
        counts the chips present and returns them in an array as
        [CR, CG, CB, CK, CW]

        sometimes has trouble detecting white chips
        """

        # blur image to reduce noise
        working_img = skimage.filters.gaussian(self.img, sigma=3/1000*self.img.shape[0], multichannel=True)

        # get the hsv channels for thresholding
        hue_img, sat_img, value_img = get_hsv_channels(working_img)

        # retrieve chips that are easily visible
        binary_color = np.logical_and(sat_img > .2, value_img > .5) # first find all bright colors
        binary_R = np.logical_and(binary_color, np.logical_or( hue_img < .10, hue_img > .90))
        binary_G = np.logical_and(binary_color, np.logical_and(hue_img > .30, hue_img < .55))
        binary_B = np.logical_and(binary_color, np.logical_and(hue_img > .55, hue_img < .70))
        binary_K = np.logical_and(np.logical_not(binary_color), value_img < .5)

        # mask the chips that have been found to avoid finding them again
        mask = np.logical_not(np.logical_or(np.logical_or(binary_R, binary_G),
                                            np.logical_or(binary_B, binary_K)))

        # the white chips suck
        img_gray = skimage.color.rgb2gray(working_img)
        plow, phigh = np.percentile(img_gray[mask], (2, 99))
        img_gray[mask] = skimage.exposure.rescale_intensity(img_gray[mask], (plow, phigh))
        binary_W = img_gray > .95

        # OK, we should have everything now
        binary_img_list = [binary_R, binary_G, binary_B, binary_K, binary_W]

        # find the edges of the patches and check if they're circles
        for img in binary_img_list:
            # binary edges
            img_edg = skimage.filters.sobel(img) > 0

            # hough transform (which is fast-ish because we only check one possible radius)
            hspace = skimage.transform.hough_circle(img_edg, [self.radii])
            accums, cx, cy, radii = skimage.transform.hough_circle_peaks(hspace, [self.radii], threshold=.25, min_ydistance=20, min_xdistance=20)

            # outputs
            self.centers.append(np.array([cy, cx]).T)
            self.nb_chips.append(len(accums))

        return self.nb_chips

    def show(self, ax=None):
        if not ax :
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        ax.imshow(self.img)
        if len(self.nb_chips) :
            ax.set_title("CR: {}, CG: {}, CB: {}, CK: {}, CW: {}".format(self.nb_chips[0], self.nb_chips[1], self.nb_chips[2], self.nb_chips[3], self.nb_chips[4]))
        else :
            ax.set_title("chips not yet counted")
        ax.axis('off')

        theta = np.linspace(0,2*np.pi, 10)
        col = ['r', 'lime', 'deepskyblue', 'k', 'w']
        for set, c in zip(self.centers, col):
            for point in set :
                ax.plot(point[1]+self.radii*np.cos(theta), point[0]+self.radii*np.sin(theta), color=c)

    def save(self, n=0):
        save_img(self.img, n, img_type='chips')


class TableSegments:
    def __init__(self, img, is_registered=False, is_equalized=False):

        tic = time.time()

        if not is_registered :
            img, corners = register_table(img)
            print("---- %.3f seconds to register table" % (time.time() - tic))
            assert len(corners), "table corners could not be detected"
            tic = time.time()

        if not is_equalized :
            img = apply_statistical_equalization(img, target_std=None)
            print("---- %.3f seconds to equalize table" % (time.time() - tic))
            tic = time.time()

        self.img = img


        # player 1
        guess = (1000/2000*img.shape[0], 1750/2000*img.shape[1])
        self.players = [PlayerSegment(1, img, guess)]

        # player 2
        guess = ( 250/2000*img.shape[0], 1425/2000*img.shape[1])
        self.players.append(PlayerSegment(2, img, guess))

        # player 3
        guess = ( 250/2000*img.shape[0],  550/2000*img.shape[1])
        self.players.append(PlayerSegment(3, img, guess))

        # player 4
        guess = (1050/2000*img.shape[0],  250/2000*img.shape[1])
        self.players.append(PlayerSegment(4, img, guess))

        print("---- %.3f seconds to extract players" % (time.time() - tic))

        # table
        tic = time.time()
        self.community = CommunitySegment(img)
        print("---- %.3f seconds to extract community cards" % (time.time() - tic))

        # chips
        tic = time.time()
        self.chips = ChipsSegment(img)
        print("---- %.3f seconds to extract chips" % (time.time() - tic))

    def evaluate(self, card_space=None) :
        """
        tries to evaluate image of the table
        """

        if not card_space :
            card_space = evaluate_cards.load_card_space("cardspace_20.pkl")

        tic = time.time()
        self.community.get_cards(card_space)
        print("---- %.3f seconds to count community cards" % (time.time() - tic))

        tic = time.time()
        for player in self.players :
            player.get_cards(card_space)
        print("---- %.3f seconds to count player cards" % (time.time() - tic))

        tic = time.time()
        self.chips.get_nb_chips()
        print("---- %.3f seconds to count chips" % (time.time() - tic))

    def show(self, title='') :
        fig = plt.figure()
        plt.suptitle(title)

        ax1 = fig.add_subplot(3,4,(1,4))
        self.chips.show(ax1)

        ax2 = fig.add_subplot(3,4,(5,8))
        self.community.show(ax2)
        # ax2.imshow(self.T)
        # ax2.set_title("T")
        # ax2.axis('off')

        for i in range(4) :
            ax = fig.add_subplot(3,4,9+i)
            self.players[i].show(ax)

        fig.tight_layout()

    def save(self, n) :
        save_img(self.img, n, img_type='table_eq')
        [player.save(n) for player in self.players]
        self.chips.save(n)

        save_img(self.T, n, img_type='cardst')


################################################################################
# testing
################################################################################

if __name__ == '__main__' :
    training_set = list(range(28)) + [99]


    # for n in [1,2,3,8,21,22] :
    for n in [0] :
    # for n in training_set:

        # get img
        start_time = time.time()
        print("\nprocessing img %d:" % n)

        tic = time.time()
        img =  get_img(n)[::2,::2,:]
        print("-- %.3f seconds to load image" % (time.time() - tic))

        tic = time.time()
        segments = TableSegments(img)
        print("-- %.3f seconds to segment image" % (time.time() - tic))

        tic = time.time()
        segments.evaluate()
        print("-- %.3f seconds to evaluate image" % (time.time() - tic))

        #
        # tic = time.time()
        # segments.save(n)
        # print("-- %.3f seconds to save images" % (time.time() - tic))

        # tic = time.time()
        # img =  get_img(n, img_type='table_eq')
        # print("-- %.3f seconds to load image" % (time.time() - tic))
        #
        # tic = time.time()
        # segments = TableSegments(img, is_registered=True, is_equalized=True)
        # print("-- %.3f seconds to segment image" % (time.time() - tic))


        segments.show("train_{}.jpg".format(str(n).zfill(2)))

        print("%.3f seconds to process image %d" % (time.time() - start_time, n))

    plt.show()
