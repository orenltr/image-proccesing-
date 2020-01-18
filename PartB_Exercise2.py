import cv2
import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
import glob
from Reader import Reader as rd
from scipy import interpolate
from PIL import Image

def transformation_parameters(ref_image_points, image2_points):
    """
    calculating parameters for projective transformation matrix
    using linear adjustment
    :param ref_image_points: homologic points in refrence image
    :param image2_points: homologic points in other image

    :type ref_image_points: np.array nx3
    :type image2_points: np.array nx3

    :return: transformation matrix
    :rtype: np.array 3x3

    """
    if (ref_image_points == image2_points).all():  # case of refrence image
        return np.eye(3)
    l = ref_image_points.reshape(-1)

    # building partial derivatives matrix
    A = np.zeros((len(l), 8))
    A[0:len(l):2, 0:3] = np.array([image2_points[:, 0], image2_points[:, 1], np.ones(int(len(l) / 2))]).T
    A[1:len(l):2, 3:6] = np.array([image2_points[:, 0], image2_points[:, 1], np.ones(int(len(l) / 2))]).T
    A[0:len(l):2, 6:8] = np.array(
        [-image2_points[:, 0] * ref_image_points[:, 0], -image2_points[:, 1] * ref_image_points[:, 0]]).T
    A[1:len(l):2, 6:8] = np.array(
        [-image2_points[:, 0] * ref_image_points[:, 1], -image2_points[:, 1] * ref_image_points[:, 1]]).T

    x = np.dot(la.inv(np.dot(A.T, A)), np.dot(A.T, l))
    x = np.append(x, 1)
    H = x.reshape((3, 3))
    return H


def bounding_rectangle(image_shape, transformation_matrix):
    """
    calculating bounding rectangle of transformed image
    :type image_shape:
    :type transformation_matrix:

    :return: bounding rectangle of transformed image
    :rtype: np.array [x_min,x_max,y_min,y_max]
    """
    x1 = np.dot(transformation_matrix, np.array([0, 0, 1]))
    x2 = np.dot(transformation_matrix, np.array([image_shape[1], 0, 1]))
    x3 = np.dot(transformation_matrix, np.array([0, image_shape[0], 1]))
    x4 = np.dot(transformation_matrix, np.array([image_shape[1], image_shape[0], 1]))

    x_min = min(x1[0], x2[0], x3[0], x4[0])
    x_max = max(x1[0], x2[0], x3[0], x4[0])
    y_min = min(x1[1], x2[1], x3[1], x4[1])
    y_max = max(x1[1], x2[1], x3[1], x4[1])

    return np.array([x_min, x_max, y_min, y_max])


def resample(original_image, grid, Hinv, trans_I):
    """
    resampling transformes image by blinear interpulation
    :param original_image: image before transformation
    :param grid: the transformed panorama frame homogenic coordinates
    :param Hinv: the invert transform matrix
    :param trans_I: panorama

    :type original_image: np.array nx3
    :type grid: np.array 3x(rxc)
    :type Hinv: np.array 3x3
    :type trans_I: np.array rxc

    :return: the panorama with the image
    :rtype: np.array rxc
    """
    # trans_image = np.zeros(grid.shape[1])
    inv_grid = np.dot(Hinv, grid)
    # computing the coordinates from the homogenic representation
    inv_grid[0] = inv_grid[0] / inv_grid[2]
    inv_grid[1] = inv_grid[1] / inv_grid[2]
    for k in range(inv_grid.shape[1]):
        j = int(np.floor(inv_grid[0, k]))  # columns integer part
        i = int(np.floor(inv_grid[1, k]))  # rows integer part
        b = inv_grid[0, k] - j  # columns fraction part
        a = inv_grid[1, k] - i  # rows fraction part

        # checking if out of image boundaries
        if i + 1 >= original_image.shape[0] or i - 1 <= 0 or j + 1 >= original_image.shape[1] or j - 1 <= 0:
            continue

        serounding_four = np.array(
            [[original_image[i, j], original_image[i, j + 1]],
             [original_image[i + 1, j], original_image[i + 1, j + 1]]])

        # interpolation
        for i in range(3):
            trans_I[int(grid[1, k]), int(grid[0, k]),i] = np.uint8(np.dot(np.array([1 - a, a]),
                                                                        np.dot(serounding_four[:, :, i],
                                                                               np.array([[1 - b], [b]]))))
    return trans_I


def build_gaussian_pyramid(image):
    G = image.copy()
    g_pyramid = [G]
    for i in range(4):
        # width = int(image.shape[1] * 0.5)
        # height = int(image.shape[0] * 0.5)
        # dim = (width, height)
        # image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        G = reduce(G)
        g_pyramid.append(G)
        # plt.figure()
        # plt.imshow(G, cmap='gray')
        # plt.xticks([]), plt.yticks([])
        # plt.show()
    return g_pyramid


def build_laplacian_pyramid(gauss_pyramid):
    l_pyramid = [gauss_pyramid[4]]
    for i in range(4, 0, -1):
        # gauss_expend = np.hstack((gauss_pyramid[i], np.array([gauss_pyramid[i][:, -1]]).T))
        # gauss_expend = np.hstack((np.array([gauss_expend[:, 0]]).T, gauss_expend))
        # gauss_expend = np.vstack((np.array([gauss_expend[0, :]]), gauss_expend))
        # gauss_expend = np.vstack((gauss_expend, np.array([gauss_expend[-1, :]])))
        gauss_expend = gauss_pyramid[i].copy()
        width = int(gauss_pyramid[i - 1].shape[1])
        height = int(gauss_pyramid[i - 1].shape[0])
        dim = (width, height)
        gauss_expend = cv2.resize(gauss_expend, dim, interpolation=cv2.INTER_AREA)
        # gauss_expend = cv2.pyrUp(gauss_pyramid[i])
        # l_pyramid.append(cv2.subtract(gauss_pyramid[i-1],gauss_expend))
        l_pyramid.append(gauss_pyramid[i - 1] - gauss_expend)

        # plt.figure()
        # plt.subplot(121)
        # # plt.imshow(l_pyramid[4 - i], cmap='gray')
        # plt.imshow(cv2.cvtColor(l_pyramid[4 - i], cv2.COLOR_BGR2RGB))
        # plt.xticks([]), plt.yticks([])
        # plt.subplot(122)
        # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        # plt.xticks([]), plt.yticks([])
        # plt.show()
    l_pyramid.reverse()
    return l_pyramid

def reduce(image):
    gaussian_filter_1d = (1/16)* np.array([[1,4,6,4,1]])
    gaussian_filter_2d = np.dot(gaussian_filter_1d.T,gaussian_filter_1d)

    image = cv2.filter2D(image,-1, gaussian_filter_2d)
    # image = image[:,0:image.shape[1]:2]
    # image = np.convolve(gaussian_filter_1d.T, image)
    image = image[0:image.shape[0]:2,0:image.shape[1]:2]
    return image

def combine_pyramids(La, Lb, Gr):
    """
    combining laplacian pyramids for image merging
    :param La: first image laplac pyramid
    :param Lb: second image laplac pyramid
    :param Gr: binary image gaussian pyramid that divides the places of the images

    :type La: np.array rxc
    :type Lb: np.array rxc
    :type Gr: np.array rxc

    :return: pyramids combination
    :rtype: np.array mxk
    """
    Ls = []
    for i in range(1, len(Gr) + 1):
        Ls.append(Gr[-i] * La[-i] + (np.ones(Gr[-i].shape) - Gr[-i]) * Lb[-i])
        # plt.figure()
        # plt.imshow(Ls[i-1], cmap='gray')
        # plt.xticks([]), plt.yticks([])
        # plt.show()
    Ls.reverse()
    return Ls
def combine_mask(image1, image2, mask):
    """
    combining images according to mask
    :param image1: first image
    :param image2: second image
    :param mask: binary image

    :type image1: np.array mxk
    :type image2: np.array mxk
    :type mask: np.array mxk (binary)

    :return: images combination
    :rtype: np.array mxk
    """
    combined = mask * image1 + (np.ones(mask.shape) - mask) * image2
    return combined


def build_from_pyramid(laplacian_pyramid):
    # laplacian_pyramid.reverse()
    image = laplacian_pyramid[4].copy()
    # for laplac_level in laplacian_pyramid[1:]:
    for i in range(3, -1, -1):
        image = np.uint8(laplacian_pyramid[i] + cv2.resize(image, (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]),
                                             interpolation=cv2.INTER_AREA))
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.xticks([]), plt.yticks([])
        # plt.subplot(122)
        # plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB), cmap='gray')
        # plt.xticks([]), plt.yticks([])
        # plt.show()
    return image

def imageSegmentation(image, thresholdValue):
    """
    image segmentation

    :param image: image
    :param thresholdValue: threshold value for segmentation

    :type image: array[256]
    :type thresholdValue: int [0, 255]

    """
    binary_image = image > thresholdValue
    return binary_image


if __name__ == '__main__':

    # importing images for panorama
    image_names = []
    image_names = glob.glob(r'panorama\*.JPG')  # reading the names of the jpg files in the folder
    images = []
    for filename in image_names:
        im = cv2.imread(filename)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        scale_percent = 20  # percent of original size
        width = int(im.shape[1] * scale_percent / 100)
        height = int(im.shape[0] * scale_percent / 100)
        dim = (width, height)
        im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

        # g = build_gaussian_pyramid(im)
        # l = build_laplacian_pyramid(g)
        # im2 = build_from_pyramid(l)
        # plt.figure()
        # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        # plt.xticks([]), plt.yticks([])
        # plt.show()

        images.append(im)
    refrence_image = images[3]

    # George = cv2.imread(r'Goerge.JPG')
    # dim = (80*2, 160*2)
    # George = cv2.resize(George, dim, interpolation=cv2.INTER_AREA)
    # gray = cv2.cvtColor(George, cv2.COLOR_BGR2GRAY)
    # thresh = imageSegmentation(gray,25)
    # mask = np.dstack((thresh, thresh, thresh))
    # mask = mask.astype(np.uint8)
    # masked = combine_mask(George, refrence_image[520*2:680*2,490*2:570*2],mask)
    # George_guass = build_gaussian_pyramid(George)
    # George_laplac = build_laplacian_pyramid(George_guass)
    # ref_gauss = build_gaussian_pyramid(refrence_image[520*2:680*2,490*2:570*2])
    # ref_laplac = build_laplacian_pyramid(ref_gauss)
    # bin_gauss = build_gaussian_pyramid(mask)
    # comb_laplac = combine_pyramids(George_laplac,ref_laplac,bin_gauss)
    # refrence_image[520*2:680*2,490*2:570*2] = build_from_pyramid(comb_laplac)
    # # refrence_image[520*2:680*2,490*2:570*2] = masked
    # plt.imshow(cv2.cvtColor(refrence_image, cv2.COLOR_BGR2RGB))
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    # importing sampled points data
    sampled_points_files = []
    sampled_points_files = glob.glob(r'panorama\*.json')  # reading the names of the json files in the folder
    images_points = []
    for filename in sampled_points_files:
        im_points = rd.ReadSampleFile(filename)
        images_points.append(im_points * (scale_percent / 100))
    refrence_image_points = images_points[3]  # points of the image that is the refrence plain

    # calculating transformation matrices
    trans_matrices = []
    for im in images_points[:4]:
        trans_matrices.append(transformation_parameters(refrence_image_points[:len(im)], im))
    trans_matrices.append(
        transformation_parameters(refrence_image_points[np.hstack([np.arange(2, 6), np.arange(7, 20)])],
                                  images_points[4]))
    trans_matrices.append(transformation_parameters(refrence_image_points[np.arange(8, 20)], images_points[5]))

    # calculating bounding rectangle of the transformed images
    bounding_recs = []
    for H in trans_matrices:
        bounding_recs.append(bounding_rectangle(refrence_image.shape, H))

    bounding_recs = np.vstack(bounding_recs)
    x_min = min(bounding_recs[:, 0])
    x_max = max(bounding_recs[:, 1])
    y_min = min(bounding_recs[:, 2])
    y_max = max(bounding_recs[:, 3])

    # calculating shift matrix T
    T = np.eye(3)
    T[0, 2] = T[0, 2] - x_min
    T[1, 2] = T[1, 2] - y_min

    # resampling
    trans_images = []
    for i in range(np.size(trans_matrices, 0)):
        # building the panorama frame
        trans_I = np.zeros((int(y_max - y_min), int(x_max - x_min), 3), dtype=np.uint8)

        # transformation metrix with shift
        trans_matrices[i] = np.dot(T, trans_matrices[i])

        # indexing for the resample
        # we resample only the the bounding rectangle of the image for efficiency
        xx, yy = np.meshgrid(range(trans_I[:, :, 0].shape[1]), range(trans_I[:, :, 0].shape[0]), indexing='ij')
        x = xx.flatten()
        y = yy.flatten()
        grid = np.vstack((x, y, np.ones(x.size)))

        # transform back to original image plain for resampling
        Hinv = la.inv(trans_matrices[i])
        trans_images.append(resample(images[i], grid, Hinv, trans_I))
        img = Image.fromarray(trans_images[i])
        # plt.figure()
        # plt.imshow(cv2.cvtColor(trans_images[i], cv2.COLOR_BGR2RGB))
        # plt.xticks([]), plt.yticks([])
        # plt.show()

    # image merging
    # building gaussian and laplacian pyramids
    gauss_pyramids = []
    pyramid_scale = 0.5     # scale between level to the next level (0-1)
    pyramid_levels_number = 4     # number of levels in the pyramid, not including base level (int)
    for im in trans_images:
        # gauss_pyramids.append(build_gaussian_pyramid(im, pyramid_scale, pyramid_levels_number))
        gauss_pyramids.append(build_gaussian_pyramid(im))

    laplac_pyramids = []
    for pyramid in gauss_pyramids:
        laplac_pyramids.append(build_laplacian_pyramid(pyramid))

    finale_panorama = trans_images[0]
    l_panorama = laplac_pyramids[1]
    binary_pyramids = []

    # combining the panorama
    for i in range(1,len(trans_images)):
        binary_image = np.zeros(finale_panorama.shape,dtype=np.uint8)
        binary_image[:,:int(bounding_recs[i,1])] = np.ones(binary_image[:,:int(bounding_recs[i,1])].shape)
        binary_pyramid = build_gaussian_pyramid(binary_image)
        # plt.figure()
        # plt.imshow(binary_image, cmap='gray')
        # plt.xticks([]), plt.yticks([])
        # plt.show()
        # g_panorama = build_gaussian_pyramid(finale_panorama)
        # l_panorama = build_laplacian_pyramid(g_panorama)
        # plt.figure()
        # plt.imshow(l_panorama[0], cmap='gray')
        # plt.xticks([]), plt.yticks([])
        # plt.show()
        l_panorama = combine_pyramids(laplac_pyramids[i],l_panorama,binary_pyramid)
        finale_panorama = build_from_pyramid(l_panorama)
        # plt.figure()
        # plt.imshow(cv2.cvtColor(finale_panorama, cv2.COLOR_BGR2RGB))
        # plt.xticks([]), plt.yticks([])
        # plt.show()

    # img = cv2.cvtColor(finale_panorama, cv2.COLOR_BGR2RGB)
    cv2.imwrite('C:/Users/Dell/Desktop/hw4-proccess/panorama20.jpg', cv2.cvtColor(finale_panorama, cv2.COLOR_BGR2RGB))
    plt.figure()
    plt.imshow(cv2.cvtColor(finale_panorama, cv2.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])
    plt.show()



    # coordinates in the panorama to put Goerge in
    s = 4
    panorama_rows = 155*s
    panorama_cols = 260*s

    # loading Goerge
    George = cv2.imread(r'Goerge.JPG')
    # finale_panorama = cv2.imread(r'panorama20.JPG')
    dim = (20*s, 40*s)
    George = np.uint8(cv2.resize(George, dim, interpolation=cv2.INTER_AREA))    # resize to fit solar panel

    # pading with zeros
    g = np.zeros(finale_panorama.shape,dtype=np.uint8)
    g[panorama_rows:panorama_rows+George.shape[0],panorama_cols:panorama_cols+George.shape[1]] = George
    George = g

    # making binary image of Goerge location by segmentation
    gray = cv2.cvtColor(George, cv2.COLOR_BGR2GRAY)
    thresh = imageSegmentation(gray,25)
    mask = np.dstack((thresh, thresh, thresh))
    mask = mask.astype(np.uint8)
    masked = combine_mask(George, finale_panorama,mask)

    # laplacian and guassian pyramids
    George_guass = build_gaussian_pyramid(George)
    George_laplac = build_laplacian_pyramid(George_guass)
    bin_gauss = build_gaussian_pyramid(mask)
    comb_laplac = combine_pyramids(George_laplac,l_panorama,bin_gauss)
    finale_panorama_pyramid = build_from_pyramid(comb_laplac)
    finale_panorama_regular = np.uint8(masked)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.imshow(cv2.cvtColor(finale_panorama_regular, cv2.COLOR_BGR2RGB))
    ax1.title.set_text('simple masking')
    ax2.imshow(cv2.cvtColor(finale_panorama_pyramid, cv2.COLOR_BGR2RGB))
    ax2.title.set_text('pyramid blending')
    plt.xticks([]), plt.yticks([])
    # ax2.xticks([]), ax2.yticks([])
    plt.show()


    # ploting pyramid example
    fig, axs = plt.subplots(5, 2)
    for i in range(len(gauss_pyramids[3])):
        axs[i,0].imshow(cv2.cvtColor(gauss_pyramids[3][i], cv2.COLOR_BGR2RGB))
        axs[i,1].imshow(cv2.cvtColor(laplac_pyramids[3][i], cv2.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])
    axs[0,0].title.set_text('gaussian pyramid')
    axs[0,1].title.set_text('laplacian pyramid')
    plt.show()
