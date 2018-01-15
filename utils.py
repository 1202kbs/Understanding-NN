import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot(samples, X_dim, channel):
	fig = plt.figure(figsize=(4, 4))
	gs = gridspec.GridSpec(4, 4)
	gs.update(wspace=0.05, hspace=0.05)

	dim1 = int(np.sqrt(X_dim))

	samples = (samples + 1) / 2

	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')

		if channel == 1:
			plt.imshow(sample.reshape([dim1, dim1]), cmap=plt.get_cmap('gray'))
		else:
			plt.imshow(sample.reshape([dim1, dim1, channel]))

	return fig


def translate(img, x, y):
	"""
	Translates the given image.

	:param x: distance to translate in the positive x-direction
	:param y: distance to translate in the positive y-direction
	:returns: the translated image as an numpy array
	"""
	M = np.float32([[1, 0, x], [0, 1, y]])
	return cv2.warpAffine(img.reshape(28,28), M, (28, 28)).reshape(1,784)


def find_roi(img, ksize, coords):
	"""
	Finds the feature with the largest relevance scores.

	:param img: the image to find the feature with the largest relevance score
	:param ksize: the size of the sliding window
	:param coords: the coordinates to ignore
	:returns: the coordinate of the feature with the largest relevance score. If the window size is larger than 1X1, function returns the position of the leftmost pixel.
	"""
	size = np.shape(img)
	temp = np.copy(img)
	for coord in coords:
		temp[coord[0]:coord[0]+ksize[0], coord[1]:coord[1]+ksize[1]] = -np.infty

	r = size[0] - ksize[0] + 1
	c = size[1] - ksize[1] + 1
	pool = [np.sum(temp[i:i+ksize[0], j:j+ksize[1]]) for i in range(r) for j in range(c)]

	return (np.argmax(pool) // c, np.argmax(pool) % c)
