import numpy as np
import matplotlib.pyplot as plt

def compute_merge_box(dets, index, base_idx, merge_idx):
    '''

    :param dets:
    :param index: sored index
    :param base_idx: the highest score bbox idx
    :param merge_idx: the bbox will be merged
    :return:
    '''

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    # calculate the points
    x11_res_min = np.min(x1[index[merge_idx]])
    x11 = np.minimum(x1[base_idx], x11_res_min)
    y11_res_min = np.min(y1[index[merge_idx]])
    y11 = np.minimum(y1[base_idx], y11_res_min)

    x22_res_max = np.max(x2[index[merge_idx]])
    x22 = np.maximum(x2[base_idx], x22_res_max)
    y22_res_max = np.max(y2[index[merge_idx]])
    y22 = np.maximum(y2[base_idx], y22_res_max)

    # print('x11: {}'.format(x11))

    return np.array([x11, y11, x22, y22, scores[base_idx]])


def merge_bbox(dets, thresh):
    '''
    merge iou > thresh bboxes
    :param dets: bboxes [[x1, y1, x2, y2, score]]
    :param thresh:
    :return:
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    # print('areas  ', areas)
    # print('scores ', scores)

    # store merge bbox
    keep = []

    # sort by score
    index = scores.argsort()[::-1]
    # print(index)

    while index.size > 0:
        # print(index.size)

        # every time the first is the biggst, and add it directly
        i = index[0]
        #
        # print('x1', x1[i])
        # print(x1[index[1:]])

        # calculate the points of overlap
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        # print('x11, y11, x22, y22')
        # print(x11, y11, x22, y22)

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h
        # print('overlaps is', overlaps)

        # compute IOUs
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        # print('ious is', ious)

        merge_idx = np.where(ious > thresh)[0]
        if len(merge_idx) > 0:
            keep.append(compute_merge_box(dets, index, i, merge_idx))
        else:
            return dets

        res_idx = np.where(ious <= thresh)[0]
        # print('res_idx {}'.format(res_idx))

        # because index start from 1
        index = index[res_idx + 1]
        # print('index : {}'.format(index))

    return np.array(keep)

def merge_bbox_overlap(dets, thresh):
    '''
    overlap / min area > thresh
    :param dets: bboxes [[x1, y1, x2, y2, score]]
    :param thresh:
    :return:
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    # print('areas  ', areas)
    # print('scores ', scores)

    # store merge bbox
    keep = []

    # sort by score
    index = scores.argsort()[::-1]
    # print(index)

    while index.size > 0:
        # print(index.size)

        # every time the first is the biggst, and add it directly
        i = index[0]
        #
        # print('x1', x1[i])
        # print(x1[index[1:]])

        # calculate the points of overlap
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        areas_min = np.minimum(areas[i], areas[index[1:]])
        # print('x11, y11, x22, y22')
        # print(x11, y11, x22, y22)

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h
        # print('overlaps is', overlaps)

        # compute overlaps / area_min
        overlaps_percent = overlaps / areas_min


        merge_idx = np.where(overlaps_percent > thresh)[0]
        if len(merge_idx) > 0:
            keep.append(compute_merge_box(dets, index, i, merge_idx))
        else:
            return dets

        res_idx = np.where(overlaps_percent <= thresh)[0]
        # print('res_idx {}'.format(res_idx))

        # because index start from 1
        index = index[res_idx + 1]
        # print('index : {}'.format(index))

    return np.array(keep)

def nms_cpu(dets, thresh):
    '''
    nms
    :param dets: bboxes [[x1, y1, x2, y2, score]]
    :param thresh:
    :return:
    '''


    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]

    keep = []

    index = scores.argsort()[::-1]

    while index.size > 0:
        # print(index.size)

        # every time the first is the biggst, and add it directly
        i = index[0]


        keep.append(i)
        # print(keep)
        # print('x1', x1[i])
        # print(x1[index[1:]])

        # calculate the points of overlap
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        # print(x11, y11, x22, y22)

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h
        # print('overlaps is', overlaps)


        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        # print('ious is', ious)


        idx = np.where(ious <= thresh)[0]
        # print(idx)

        # because index start from 1
        index = index[idx + 1]
        # print(index)
    return dets[keep]


def plot_bbox(dets, c='k'):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    plt.plot([x1, x2], [y1, y1], c)
    plt.plot([x1, x1], [y1, y2], c)
    plt.plot([x1, x2], [y2, y2], c)
    plt.plot([x2, x2], [y1, y2], c)
    plt.title(" merge boxes")


if __name__ == '__main__':
    test_array = np.array([[2.92830582e+01, 1.79133560e+02, 1.02217841e+03, 7.46682129e+02,
        9.42870200e-01],
       [1.42071198e+02, 1.79247421e+02, 1.22399792e+03, 7.56439575e+02,
        9.89860892e-01],
       [1.04468565e+01, 1.75340271e+02, 1.20489661e+03, 7.48874878e+02,
        9.99264896e-01],
       [4.14248924e+01, 1.82301193e+02, 7.56095886e+02, 7.46787292e+02,
        8.38867903e-01],
       [3.96124001e+01, 1.82976425e+02, 1.13396570e+03, 7.51773438e+02,
        2.76165903e-01],
       [2.27845840e+02, 1.74224060e+02, 1.23750378e+03, 7.43472046e+02,
        4.32294488e-01],
       [2.80735035e+01, 1.72266327e+02, 9.24836182e+02, 7.44747925e+02,
        3.61437321e-01],
       [1.44738937e+02, 1.79717041e+02, 7.93623779e+02, 7.49015930e+02,
        1.63943082e-01],
       [4.62852112e+02, 1.84467819e+02, 1.11693933e+03, 7.41876953e+02,
        3.61700803e-01]])
    print('input shape {}'.format(test_array.shape))
    keep = merge_bbox(test_array, 0.6)
    print(keep)

    plt.figure(1)
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    plt.sca(ax1)
    plt.sca(test_array, 'k')

    plt.sca(ax2)
    plot_bbox(keep, 'r')