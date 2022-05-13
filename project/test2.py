#!/usr/bin/env python3.8

import time
from cycler import cycler
import matplotlib

from test import *

def sigmoid(x) :
    return 1/(1+np.exp(-4*x + 2))

m = 0

for n in [0,1,8,11,21,22] :
# for n in [15] :
# for n in training_set :
    print("img", n)


    # load in the registered table
    tic = time.time()
    table = get_img(n, img_type='table')
    print("--- %.3f seconds to load table ---" % (time.time() - tic))

    tic = time.time()
    table_eq = apply_statistical_rescale(table, target_std=None)#, range_type='clip')
    print("--- %.3f seconds to equalize table ---" % (time.time() - tic))
    del table
    # show(table_eq, "img {}".format(n))
    # save_img(table_eq, n, "table_eq")

    # load in the equalized table
    # tic = time.time()
    # table = get_img(n, img_type='table_eq')
    # print("--- %.3f seconds to load table ---" % (time.time() - tic))

    # show(table_eq, str(n))

    # tidx = skimage.color.rgb2gray(table_eq) < 0.65
    # # tidx = skimage.morphology.binary_closing(tidx,  skimage.morphology.disk(3))
    # table[tidx] = 0

    # table_eq[skimage.filters.gaussian(skimage.color.rgb2gray(table_eq) < 0.80] = 0
    # show(table_eq, str(n))

    img_edg = skimage.feature.canny(skimage.color.rgb2gray(table_eq), sigma=1.5)
    show(img_edg, str(n))

plt.show()
