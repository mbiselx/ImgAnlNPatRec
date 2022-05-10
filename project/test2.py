#!/usr/bin/env python3.8

from test import *
import time


# for n in training_set :
#     print("img", n)
#     table = get_img(n, img_type='table_eq')
#     save_img(skimage.color.rgb2hsv(table), n, img_type="table_hsv")
# exit()



m = 0
# for n in [1,2,3,8,21,22] :
# for n in [1] :
for n in training_set :
    print("img", n)


    # load in the registered table
    tic = time.time()
    table = get_img(n, img_type='table')
    print("--- %.3f seconds to load table ---" % (time.time() - tic))

    # segment the teable
    tic = time.time()
    segments = segment_table_simple(table)
    # segments = segment_table(table)
    print("--- %.3f seconds to segment table ---" % (time.time() - tic))

    for player in segments.player :
        if player.has_folded :
            continue
        img_canny = skimage.feature.canny(skimage.color.rgb2gray(player.img), sigma=1)
        # _ , angles, dists = skimage.transform.hough_line_peaks(*skimage.transform.hough_line(img_canny), num_peaks=4)
        # # get all intersections
        # intersects = []
        # for i in range(len(angles)-1) :
        #     [intersects.append(get_intersect(dists[i],angles[i], d,a)) for d, a in zip(dists[i+1:], angles[i+1:])]
        # intersects = np.array(list(filter(np.any, intersects)))
        # # only take intersections in image area
        # intersects = intersects[np.logical_and(np.logical_and(intersects[:,0] > 0, intersects[:,0] < player.img.shape[1]),
        #                                        np.logical_and(intersects[:,1] > 0, intersects[:,1] < player.img.shape[0]))]

        # show(img_canny)
        # plt.scatter(intersects[:,0], intersects[:,1])
        # all_cont  = skimage.measure.find_contours(img, 0.1*img.max())
        # long_cont = max(all_cont, key=len)
        # p.show()
        # plt.plot(long_cont[:, 1], long_cont[:, 0], color='g', linewidth=2)

plt.tight_layout()
plt.show()
