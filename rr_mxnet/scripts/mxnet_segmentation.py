#!/usr/bin/env python

import mxnet
import numpy as np
import gluoncv as gcv
import mxnet as mx
import cv2

# MXNet-based semantic segmentation
# models: fcn_resnet50_ade, psp_resnet50_ade, deeplab_resnet50_ade
# we want to make a mask where we want detection results to be valid (mask out water, sky, treetops, etc), just grass plus some vertical boundary extent
# classes in ADE20k: people=12, sky=2, grass=9, road=6, water=21, rock=34. Generally for the goose detector, we might want only grassy and road-like areas.
# for driveable surface detector, we might instead want known, problematic areas like water, lake, sea, rock, etc
class MxNetSegmentation(object):
    def __init__(self, model_directory, model_filename, network_name='deeplab_resnet50_ade', gpu_enabled=True, image_resize=300):
        # model settings
        self.prefix = str(model_directory) + '/' + str(model_filename)
        self.network_name = network_name
        self.image_resize=image_resize
        if gpu_enabled:
            self.ctx = mxnet.gpu(0)
        else:
            self.ctx = mxnet.cpu()

        self._mean=(0.485, 0.456, 0.406)
        self._std=(0.229, 0.224, 0.225)

        # Create Detector
        self.net = gcv.model_zoo.get_model(self.network_name, pretrained=True)

    def segment(self, image):
        orig_image_size=image.shape

        # set image_size to that desired by model
        image_size=self.image_resize

        data = mxnet.nd.array(image)
        data = mx.image.imresize(data, self.image_resize, self.image_resize)
        data = mx.nd.image.to_tensor(data)
        data = mx.nd.image.normalize(data, mean=self._mean, std=self._std)
        data = data.expand_dims(0).as_in_context(self.ctx)

        output = self.net.forward(data)
        segmentation = mx.nd.squeeze(mx.nd.argmax(output[0], 1))

        # resize to original image size
        segmentation = cv2.resize(segmentation.asnumpy(), (orig_image_size[1],orig_image_size[0]), interpolation=cv2.INTER_NEAREST)

        # return segmentation
        return segmentation
    

def convert_frame_to_jpeg_string(frame):
    return np.array(cv2.imencode('.jpg', frame[:,:,[2,1,0]])[1]).tostring()

