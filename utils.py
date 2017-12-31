import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

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