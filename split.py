import os
import timeit
import numpy as np
import numba
import cv2


def ImageCrops(img, dim):
    img = ImageResize(img, dim)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    crops = ImageDivision3(img, dim)
    return crops

def ImageCrops_3d(img, dim):
    img = ImageResize(img, dim)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    crops = ImageDivision(img, dim)
    return crops

def ImageResize(img, dim):
    return cv2.resize(img, (int(np.floor(img.shape[1] / dim[1]) * dim[1]), int(np.floor(img.shape[0] / dim[0]) * dim[0])))

def ImageReshape(img, dim):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pad_right = int(np.floor(img.shape[1] / dim[1]) * dim[1]) - img.shape[1]
    pad_down = int(np.floor(img.shape[0] / dim[0]) * dim[0]) - img.shape[0]
    np.pad(img, ([0, pad_down], [0, pad_right]))
    return np.pad(img, ([0, pad_down], [0, pad_right]))

# @numba.jit(cache=True)
def ImageDivision(img, dim):
    # Divide image into blocks with shape (dim, dim, d)

    # Find height and width and depth of image
    h, w, d = img.shape

    # Find number of blocks in x-axis and y-axis
    nx = int(np.floor(h / dim[0]))
    ny = int(np.floor(w / dim[1]))

    blocks = np.lib.stride_tricks.as_strided(img, (nx, ny, dim[0], dim[1], d),
                                             (img.strides[0] * dim[0], img.strides[1] * dim[1], img.strides[0],
                                              img.strides[1], img.strides[2]))
    return blocks


# @numba.jit(cache=True)
def ImageDivision2(img, dim):
    # Divide image into blocks with shape (dim, dim, d)

    # Find height and width and depth of image
    h, w = img.shape

    # Find number of blocks in x-axis and y-axis
    nx = int(np.floor(h / dim))
    ny = int(np.floor(w / dim))

    blocks = np.swapaxes(img.reshape(nx, dim, ny, -1), 1, 2)
    return blocks


# @numba.jit(cache=True)
def ImageDivision3(img, dim):
    # Divide image into blocks with shape (dim, dim)

    # Find height and width of image
    h, w = img.shape

    # Find number of blocks in x-axis and y-axis
    nx = int(np.floor(h / dim[0]))
    ny = int(np.floor(w / dim[1]))

    # Find each block and return as 4d array with shape (nx, ny, dim, dim)
    blocks = np.lib.stride_tricks.as_strided(img, (nx, ny, dim[0], dim[1], 1),
                                             (img.strides[0] * dim[0], img.strides[1] * dim[1], img.strides[0],
                                              img.strides[1], 0))
    return blocks


def reshape(blocks):
    return np.reshape(blocks, (blocks.shape[0] * blocks.shape[1], blocks.shape[2], blocks.shape[3], blocks.shape[4]))

SIZE = (224, 224)

if __name__ == '__main__':
    imgFolder = '/home/reyhane/PycharmProjects/trainApp_oxin8/oxin_image_grabber/995/BOTTOM/1'
    # imgFolder = '/home/dorsa/TrainApp/trainApp_oxin/oxin_image_grabber/11/TOP/4'

    # List files in folder
    imgFiles = os.listdir(imgFolder)
    # imgs = []

    for file in imgFiles:
        # Read images
        img = cv2.imread(os.path.join(imgFolder, file), 0)
        start = timeit.default_timer()
        crops = ImageCrops(img, SIZE)
        end = timeit.default_timer()
        print((end - start) * 1000)

        # cv2.imshow('o', img)
        # for i in range(crops.shape[0]):
        #     for j in range(crops.shape[1]):
        #         cv2.imshow('', crops[i, j])
        #         cv2.waitKey(0)

        # img = cv2.resize(img, (1920, 1200))
        # imgs.append(img)

    # ImageCrops(np.ones((1200, 1920), dtype=np.uint8), SIZE)

    # start = timeit.default_timer()
    # crops = []
    # for i in range(len(imgs)):
    #     img = imgs[i]
    #     crops.append(ImageCrops2(img, SIZE))
    #
    # result = np.stack(crops, axis=0).reshape(
    #     (len(crops), crops[0].shape[0]*crops[0].shape[1], SIZE[0], SIZE[1], 1))
    # end = timeit.default_timer()
    # print((end - start) * 1000)
    # # for i in range(result.shape[0]):
    # #     for j in range(result.shape[1]):
    # #         cv2.imshow('', result[i, j])
    # #         cv2.waitKey(0)
    # #
