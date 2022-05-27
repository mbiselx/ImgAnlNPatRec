#!/usr/bin/env python3.8
import pickle
from preprocessing import *


def get_relevant_contours(img, ll=65, lh=300) :
    """
    get the contours from an image which could represent a number
    """
    img_to_cont = img if len(img.shape) < 3 else skimage.color.rgb2gray(img)
    #img_to_cont =  skimage.filters.gaussian(img_to_cont, sigma=.5)
    # img_to_cont =  skimage.filters.unsharp_mask(img_to_cont)
    contours = skimage.measure.find_contours(img_to_cont, 0.7) +                \
               skimage.measure.find_contours(img_to_cont, 0.8) +                \
               skimage.measure.find_contours(img_to_cont, 0.9)

    return filter(lambda cont : (len(cont) > ll and len(cont) < lh), contours)  # only return the relevant contours


def get_fourier_descriptors(contour, nb_descr=2, descr_idx=[], make_contour=False) :
    """
    gets the fourier decriptors of a controur. Preserves the relative phases
    """
    # step 1 : fourier transform of the contour
    fft_cont = np.fft.fft(contour[:,0] + contour[:,1]*1j)

    # step 2 : get the interesting indices
    if len(descr_idx) :                                                         # if indices of descriptors are specified
        descr_idx = np.append(descr_idx, 0)                                     # add the idx '0' for the contour aproximation
    else :                                                                      # otherwise take the nb_descr lowest frequencies
        descr_idx = np.arange(-np.floor(nb_descr/2), np.ceil(nb_descr/2)+1, dtype=int)

    # step 3 : extract the location, rotation and relevant descriptors
    location    = np.array([fft_cont[0].real, fft_cont[0].imag])/fft_cont.shape[0] # get the location of the center
    # rotation    = np.exp(-1j*(np.angle(fft_cont[1]) + np.angle(fft_cont[0])))                             # rotation is given by the phase of the non-DC component
    rotation    = np.exp(-1j*np.angle(fft_cont[1]))
    # rotation    = 1
    descriptors = fft_cont[descr_idx[descr_idx!=0]]                             # remove the idx '0' (location)

    # step 4 : normalize descriptors onto the unit ball, but keep their relative phases
    # descriptors = descriptors * rotation                                        #  only correct for initial phase
    # descriptors = np.abs(descriptors) / np.linalg.norm(descriptors)             # normalize over descriptors and disregard phase
    # descriptors = descriptors / np.linalg.norm(descriptors)                     # normalize over descriptors
    # descriptors = descriptors / np.linalg.norm(fft_cont[1:])                    # normalize over the entire fft
    descriptors = descriptors / np.linalg.norm(descriptors) * rotation          # normalize over descriptors and correct for initial phase
    # descriptors = descriptors / np.linalg.norm(fft_cont[1:]) * rotation         # normalize over the entire fft and correct for initial phase

    # step 5 : visualization ?
    if not make_contour :
        return location, descriptors, rotation
    else :
        aprx_cont = np.zeros_like(fft_cont)       # make a contour approximation for visualization
        aprx_cont[descr_idx] = fft_cont[descr_idx];
        aprx_cont = np.fft.ifft(aprx_cont)
        aprx_cont = np.array([aprx_cont.real, aprx_cont.imag]).T
        return location, descriptors, aprx_cont, rotation


def make_avg_descr(card, nb_descr=10, make_contour=False) :
    """
    creates average descriptors out of all available images for a card value
    """

    nb_imgs = len(os.listdir(os.path.join(os.getcwd(), 'data', card)))

    avg_descr = [[] for i in range(len(card))]
    for n in range(nb_imgs) :
        img = get_img(n, card)
        for contour in get_relevant_contours(img, ll=75) :
            loc, descriptors, _ = get_fourier_descriptors(contour, nb_descr)
            # descriptors = np.abs(descriptors)
            i = int((loc[1]*len(card))//img.shape[1])
            avg_descr[i].append(descriptors)

    avg_descr = dict([(char, np.array(avg_descr[i]).mean(0)) for i, char in enumerate(card)])

    if not make_contour :
        return avg_descr
    else :
        avg_cont = dict()
        for char in card :
            avg_cont_fft = np.zeros(100, dtype=np.complex);
            avg_cont_fft[1:(1+nb_descr//2)] = avg_descr[char][nb_descr//2:]
            avg_cont_fft[-nb_descr//2:] = avg_descr[char][:nb_descr//2]
            avg_char_cont = np.fft.ifft(avg_cont_fft)
            avg_cont[char] = np.array([avg_char_cont.real, avg_char_cont.imag]).T
        return avg_descr, avg_cont

def make_card_space(cards = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "C", "D", "H", "S"], nb_descr=10) :
    """
    creates a dictionary containing the centroids of each card value
    """

    card_space   = dict()
    [card_space.update(make_avg_descr(card, nb_descr)) for card in cards]
    return card_space

def show_card_space(card_space) :
    """
    show the centroids of the card space
    """
    nb_descr = len(card_space[list(card_space.keys())[0]])

    fig, axes = plt.subplots(4, 5)
    fig.suptitle("card space for nb_descr = {}".format(nb_descr))

    for card, ax in zip(card_space.keys(), axes.ravel()[:len(card_space.keys())]) :
        avg_cont_fft = np.zeros(100, dtype=np.complex);
        avg_cont_fft[1:(1+nb_descr//2)] = card_space[card][nb_descr//2:]
        avg_cont_fft[-nb_descr//2:] = card_space[card][:nb_descr//2]
        avg_cont = np.fft.ifft(avg_cont_fft)
        avg_cont = np.array([avg_cont.real, avg_cont.imag]).T

        ax.plot(avg_cont[:,1], -avg_cont[:,0])
        ax.axis('image')
        ax.set_title(card)

    [ax.axis('off') for ax in axes.ravel()]

def save_card_space(card_space, filename="cardspace.pkl") :
    """
    dump the card-space to a PKL file
    """
    save_path = os.path.join(os.getcwd(), 'data', 'models')
    if not os.path.exists(save_path) :
        os.makedirs(save_path)

    file_path = os.path.join(save_path, filename)
    file = open(file_path, "wb")
    pickle.dump(card_space, file)
    file.close()

def load_card_space(filename="cardspace.pkl") :
    """
    read a card-space from a PKL file
    """
    file_path = os.path.join(os.getcwd(), 'data', 'models', filename)
    file = open(file_path, "rb")
    card_space = pickle.load(file)
    file.close()
    return card_space

def detect_card_symbols(img, card_space, thresh=0.15, min_dist=20, nb_vals=None) :
    """
    tries to detect all the card symbols in a given image
    """
    detected_nums = []
    nb_descr = len(card_space[list(card_space.keys())[0]])
    for contour in get_relevant_contours(skimage.filters.unsharp_mask(img, amount=.3, multichannel=True)) :
        location, descriptors, _ = get_fourier_descriptors(contour, nb_descr)

        closest_num = sorted([(np.linalg.norm(np.abs(descriptors) - np.abs(card_space[num])), num) for num in card_space])[0]
        # closest_num = sorted([(np.linalg.norm(descriptors - card_space[num]), num) for num in card_space])[0]
        if closest_num[0] < thresh :                                            # if the cardspace distance is small enough
            detected_nums.append((closest_num[1], location, closest_num[0]))    # (card_value, location, cardspace_distance)


    # special case : 10 is made up of two characters, which must be fused
    to_remove = set()
    iterator = enumerate(detected_nums)
    for i, num1 in iterator:
        if num1[0] == '0' :
            to_remove.add(i)                                                    # remove all 0's
        if num1[0] == '1' :
            to_remove.add(i)                                                    # remove all 1's
            for j, num2 in enumerate(detected_nums) :
                if i != j and num2[0] == '0':
                    if np.linalg.norm(num1[1] - num2[1]) < min_dist :
                        detected_nums.append(('10', .5*(num1[1]+num2[1]), .5*(num1[2]+num2[2])))   # all 1's associated with 0's are combined to 10's
    detected_nums = [v for i, v in enumerate(detected_nums) if i not in to_remove]

    # remove duplicate or overlapping detections
    to_remove = set()
    for i, num1 in enumerate(detected_nums) :
        for j, num2 in enumerate(detected_nums)  :
            if i == j : break
            if np.linalg.norm(num1[1] - num2[1]) < min_dist :         # compare locations (only one number can be in a single spot)
                to_remove.add(i if num1[2] > num2[2] else j)                    # compare cardspace_distance (chose the most likely number)
    detected_nums = [v for i, v in enumerate(detected_nums) if i not in to_remove]

    #only return the most likely detections
    detected_nums = sorted(detected_nums, key=lambda val : val[2])

    if nb_vals :
        return detected_nums[:nb_vals]
    else :
        return detected_nums

def guess_cards(img, card_space, expected_cards, disp=False, title='') :
    """
    tries to guess what cards are present in an image
    """

    # container to store detected cards
    detected_cards = []

    # detect all card-space symbols present in the image
    card_symbols = detect_card_symbols(img, card_space)

    # sort detected symbols into suits and values
    suits = [sym for sym in card_symbols if sym[0] in ['C', 'D', 'H', 'S']  ]
    vals  = [sym for sym in card_symbols if sym[0] not in ['C', 'D', 'H', 'S']  ]

    # associate each value with a suit, if possible
    to_remove_v, to_remove_s = set(), set()
    iterator = list(enumerate(vals)) # make iterator before looping
    for i, v in iterator :
        ss = [(s, j) for j, s in enumerate(suits) if j not in to_remove_s and (np.linalg.norm(v[1] - s[1]) < 40)]
        if len(ss) :
            s, j = min(ss, key=lambda elem: elem[0][2])
            loc = .5*(v[1] + s[1])
            lik = .5*(v[2] + s[2])
            vals.append((v[0] + s[0], loc, lik))
            to_remove_v.add(i)
            to_remove_s.add(j)
    vals  = [v for i, v in enumerate(vals)  if i not in to_remove_v]
    suits = [s for j, s in enumerate(suits) if j not in to_remove_s]

    # first try to detect the cards that are fully visible
    to_remove_v, to_remove_s = set(), set()
    iterator = list(enumerate(vals))
    for i, v1 in iterator :
        for j, v2 in iterator :
            if i == j : break
            if (i in to_remove_v or j in to_remove_v) : continue

            if np.linalg.norm(v2[1] - v1[1]) < 400 :
                # spacial case 1 : we've detected a specific card
                # OR
                # spacial case 2 : we have a detected a similar value, but no suit
                # OR
                # spacial case 3 : confusion between 6 and 9
                intersect = ''.join([c for c in v1[0] if c in v2[0]])
                if (intersect == v1[0] and intersect == v2[0]) or               \
                   (len(intersect) and intersect[-1] not in ['C', 'D', 'H', 'S']):
                    loc = .5*(v1[1] + v2[1])
                    if loc[0] < img.shape[0] * 1./3. or loc[0] > img.shape[0] * 2./3. :
                        break # the card location doesn't make sense
                    val = intersect
                    lik = .5*(v1[2] + v2[2])
                    if val[-1] not in ['C', 'D', 'H', 'S'] :
                        ss = [(s, k) for k, s in enumerate(suits) if k not in to_remove_s and (np.linalg.norm(v1[1] - s[1]) < 60 or np.linalg.norm(v2[1] - s[1]) < 60)]
                        s, k = min(ss, key=lambda elem: elem[0][2]) if len(ss) else ('', None)
                        if s :
                            to_remove_s.add(k)
                            val = val + s[0]
                            lik = .5*(v1[2] + v2[2] + s[2])
                    to_remove_v.add(i)
                    to_remove_v.add(j)
                    detected_cards.append((val, loc, lik))
    vals  = [v for i, v in enumerate(vals)  if i not in to_remove_v]
    suits = [s for j, s in enumerate(suits) if j not in to_remove_s]
    detected_cards = sorted(detected_cards + vals, key=lambda val : val[2])
    detected_cards = detected_cards[:expected_cards]





    if disp :
        show_img(img, title)

        for card in detected_cards :
            plt.text(card[1][1]+10, card[1][0]+10, card[0], color='limegreen', size=20)

        for v in vals :
            plt.text(v[1][1]+10, v[1][0]+10, v[0], color='dodgerblue', size=13)
        for s in suits :
            plt.text(s[1][1]+10, s[1][0]+10, s[0], color='deepskyblue', size=12)

    return detected_cards

if __name__ == '__main__':

    nb_descr = 20
    # for nb_descr in range(1,11) :
    #     card_space = make_card_space(nb_descr=10*nb_descr)
    #     save_card_space(card_space, f"cardspace_{10*nb_descr}.pkl")
    # exit()

    # card_space = load_card_space(f"cardspace_{nb_descr}.pkl")
    # show_card_space(card_space)
    # plt.show()

    card_space = make_card_space(nb_descr=nb_descr)
    # show_card_space(card_space)
    # plt.show()

    # cards_list = [4, 14, 17, 18] #[0, 2, 3, 4, 5, 7, 8, 9]
    # cards_name = 'cardst'
    # for n in cards_list :
    #     img = get_img(n, cards_name)
    #
    #     # print results
    #     guess_cards(img, card_space, expected_cards=5, disp=True, title="{}_{}".format(cards_name, str(n).zfill(2)))
    # plt.show()

    cards_list = [0, 2, 3, 4, 5, 7, 8, 9]
    cards_name = 'cards1'
    for n in cards_list :
        img = get_img(n, cards_name)

        # print results
        guess_cards(img, card_space, expected_cards=2, disp=True, title="{}_{}".format(cards_name, str(n).zfill(2)))
    plt.show()
