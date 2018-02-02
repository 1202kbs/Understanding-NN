import tensorflow as tf
import numpy as np


class Grad:
    
    def __init__(self, sess, nodes, img_size, channel):
        '''
        Initialize the Grad class.
        
        :param sess: session on which the computation graph is built
        :param nodes: list containing the name, tensor, or operation for image input and softmax/logit output [input, output]
        Assumes usage of tf.placeholder_with_default() for other training hyperparameters
        :param img_size: list indicating the size of image [width, height]
        :param channel: number of channels of the image. 1 for grayscale and 3 for rgb
        '''
        
        self.sess = sess
        
        self.nodes = []
        for node in nodes:
            
            if type(node) is str:
                
                if node[-1] != 0:    
                    self.nodes.append(self.sess.graph.get_tensor_by_name(node + ':0'))
                else:
                    self.nodes.append(self.sess.graph.get_tensor_by_name(node))
            
            elif type(node) is tf.Tensor:
                self.nodes.append(node)
            elif type(node) is tf.Operation:
                self.nodes.append(self.sess.graph.get_tensor_by_name(node.name + ':0'))
            else:
                raise Exception('Node must either be type of {}, {}, or {}.'.format(str, tf.Tensor, tf.Operation))
        
        self.img_size = img_size
        self.channel = channel
                
    def reshape(self, imgs):
        '''
        Reshape the given image to fit the shape required by the input tensor.
        
        :param imgs: images to reshape
        :returns: reshaped images
        '''
        
        shape = self.nodes[0].get_shape().as_list()
        shape[0] = -1
        return np.reshape(imgs, shape)
    
    def resize(self, imgs):
        '''
        Resizes the given images to take the shape of [num_images, width, height, channel].
        
        :param imgs: images to resize
        :returns: resized images
        '''
        return np.reshape(imgs, [-1, self.img_size[0], self.img_size[1], self.channel])
    
    def inference(self, imgs, argmax=True):
        '''
        Performs inference on the given images.
        
        :param imgs: images to perform inference on
        :param argmax: if True, returns the indices of max logits; else, returns the logits themselves
        :returns: result of inference
        '''
        
        imgs = self.reshape(imgs)
        res = self.sess.run(self.nodes[1], feed_dict={self.nodes[0]: imgs})
        
        if argmax:
            return np.argmax(res, axis=1)
        else:
            return res
    
    def gradient(self, imgs, inds=None):
        '''
        Calculates the gradient of given images with respect to logits.
        
        :param imgs: images to calculate gradient of
        :param inds: expects an array/list of indices of logits to calculate gradients for.
        If None is given, use max logits from inference
        :returns: gradients
        '''
        
        imgs = self.reshape(imgs)
        
        if not inds:
            inds = self.inference(imgs)
        
        logs = [[i, inds[i]] for i in range(len(imgs))]
        res = self.sess.run(tf.gradients(tf.gather_nd(self.nodes[1], logs), self.nodes[0])[0], feed_dict={self.nodes[0]: imgs})
        
        return self.resize(res)
        
    def smooth_grad(self, imgs, inds=None, noise_level=None, sample_size=None, interval=None):
        '''
        Calculates SmoothGrad of given images.
        
        :param imgs: images to calculate SmoothGrad of
        :param inds: logit indices to calculate SmoothGrad with respect to
        :param noise_level: standard deviation of random normal noise. Calculated by (max_val - min_val) * noise_level
        :param sample_size: number of samples to calculate average gradient
        :param interval: interval to select smooth grad images
        :returns: SmoothGrad results
        '''
        res = []
        
        imgs = self.resize(imgs)
        
        if not inds:
            inds = self.inference(imgs)
        if not noise_level:
            noise_level = 0.1
        if not sample_size:
            sample_size = 50
        if not interval:
            interval = 1
        
        for i in range(len(imgs)):
            img = imgs[None,i]
            sigma = (np.max(img) - np.min(img)) * noise_level
            noise_imgs = [img + np.random.normal(scale=sigma, size=np.shape(img)) for i in range(sample_size)]
            grads = self.gradient(noise_imgs, [inds[i]] * sample_size)
            
            temp = [grads[0]]
            
            for j in range(sample_size - 1):
                temp.append(temp[-1] + grads[j + 1])
            
            temp = temp[::interval] / np.reshape(np.arange(0,sample_size,interval) + 1, [-1] + [1] * 3)
            res.append(temp)

        return np.array(res)
    
    def integrated_grad(self, imgs, inds=None, steps=None, use_smooth=None, noise_level=None, sample_size=None):
        '''
        Calculates Integrated Gradients of given images.
        
        :param imgs: images to calculated Integrated Gradients of
        :param inds: indices to calculate Integrated Gradients with respect to
        :param steps: number of steps to perform Riemann Sum
        :param use_smooth: indicate whether to use SmoothGrad along with Integrated Gradients
        :param noise_level: standard deviation of random normal noise when using SmoothGrad.
        Calculated by (max_val - min_val) * noise_level
        :param sample_size: number of samples to calculate average gradient when using SmoothGrad
        :returns: Integrated Gradient results
        '''
        res = []
        
        imgs = self.resize(imgs)
        
        if not inds:
            inds = self.inference(imgs)
        if not use_smooth:
            use_smooth=False
        if not steps:
            steps = 50
        
        for i in range(len(imgs)):
            img = imgs[None,i]
            scaled_imgs = [(float(i) / steps) * img for i in range(1, steps + 1)]
            
            if use_smooth:
                grads = self.smooth_grad(scaled_imgs, [inds[i]] * steps, noise_level, sample_size)[:, -1]
            else:
                grads = self.gradient(scaled_imgs, [inds[i]] * steps)
            
            res.append(img * np.average(grads, axis=0))
        
        return self.resize(res)
    
    def truncate(self, attrs, ptile):
        '''
        Truncate given attribute map to indicated percentile.
        
        :param attrs: attribute maps
        :param ptile: percentile
        :returns: truncated attribute maps
        '''
        res = []
        
        attrs = abs(attrs)
        
        for i in range(len(attrs)):
            temp = np.clip(attrs[i] / np.percentile(attrs[i], ptile), 0, 1)      
            res.append(temp)
            
        return np.array(res)
    
    def grayscale(self, attrs):
        '''
        Converts RGB attribute maps to grayscale. If channel is 1, nothing happens.
        
        :param attrs: attribute maps
        :returns: grayscale attribute maps
        '''
        
        if self.channel == 1:
            return attrs
        
        res = []
        
        for i in range(len(attrs)):
            temp = np.average(attrs[i], axis=2)
            temp = np.transpose([temp] * 3, axes=[1, 2, 0])
            res.append(temp)
        
        return np.array(res)
    
    def visualize_attrs(self, imgs, gradient_type='gradient', inds=None, noise_level=None, sample_size=None, interval=None, use_smooth=None, steps=None, ptile=None):
        '''
        Calculates imgs * attribute maps.
        
        :param imgs: images
        :param gradient_type: type of gradient to use.
        'gradient' for vanilla gradient, 'smooth' for SmoothGrad, and 'integrated' for Integrated Gradients.
        :returns: imgs * attribute maps
        '''
        
        res = []
        
        imgs = self.resize(imgs)
        
        if not ptile:
            ptile = 99
        
        if gradient_type == 'gradient':
            attrs = self.gradient(imgs, inds)
        elif gradient_type == 'smooth':
            attrs = self.smooth_grad(imgs, inds, noise_level, sample_size, interval)
        elif gradient_type == 'integrated':
            attrs = self.integrated_grad(imgs, inds, steps, use_smooth, noise_level, sample_size)
        else:
            raise Exception('Unknown type of gradient technique. Must either be type of "gradient", "smooth", or "integrated".')
        
        if self.channel == 1:
            
            if gradient_type == 'smooth':
                orig_shape = np.shape(attrs)
                attrs = self.resize(attrs)
                attrs = self.truncate(attrs)
                attrs = np.reshape(attrs, orig_shape)
                return np.expand_dims(imgs, axis=1) * self.truncate(attrs, ptile)
            else:
                return imgs * self.truncate(attrs, ptile)
        
        else:
            
            if gradient_type == 'smooth':
                orig_shape = np.shape(attrs)
                attrs = self.resize(attrs)
                attrs = self.grayscale(attrs)
                attrs = self.truncate(attrs)
                attrs = self.reshape(attrs, orig_shape)
                return np.expand_dims(imgs, axis=1) * self.truncate(attrs, ptile)
            else:
                return imgs * self.truncate(attrs, ptile)