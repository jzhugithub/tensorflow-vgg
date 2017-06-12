import skimage
import skimage.io
import skimage.transform
import numpy as np


# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image_cope(path, edge = 224):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0.0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    return skimage.transform.resize(crop_img, (edge, edge))


def load_image_fill(path, edge = 224):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0.0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    fill_img = np.zeros([edge, edge, img.shape[2]])
    if img.shape[0] > img.shape[1]:
        short_edge = edge * img.shape[1] / img.shape[0]
        fill_img[:, (edge - short_edge) / 2:(edge + short_edge) / 2] = skimage.transform.resize(img, (edge, short_edge))
    else:
        short_edge = edge * img.shape[0] / img.shape[1]
        fill_img[(edge - short_edge) / 2:(edge + short_edge) / 2, :] = skimage.transform.resize(img, (short_edge, edge))
    return fill_img


def load_image_resize(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = 224
        nx = 224
    return skimage.transform.resize(img, (ny, nx))


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1




def test():
    img = skimage.io.imread("./test_data/tiger.jpeg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    skimage.io.imshow(img)
    skimage.io.show()
    skimage.io.imsave("./test_data/output.jpg", img)


if __name__ == "__main__":
    # test()

    img = load_image_fill("./test_data/puzzle.jpeg")
    skimage.io.imshow(img)
    skimage.io.show()
