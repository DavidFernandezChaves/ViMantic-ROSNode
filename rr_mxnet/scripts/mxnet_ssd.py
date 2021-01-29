#!/usr/bin/env python

import mxnet
import numpy as np
import gluoncv as gcv
import mxnet as mx

# MXNet-based single-shot detector 
class MxNetSSDClassifier(object):
    def __init__(self, model_directory, model_filename, network_name='ssd_512_resnet50_v1_coco', batch_size=1, gpu_enabled=True, num_classes=20, downsample_size=512):
        # model settings
        self.prefix = str(model_directory) + '/' + str(model_filename)
        self.num_classes = num_classes
        self.batch_size=batch_size
        self.network_name = network_name
        self.image_size=downsample_size
        if gpu_enabled:
            self.ctx = mxnet.gpu(0)
        else:
            self.ctx = mxnet.cpu()

        self._mean=(0.485, 0.456, 0.406)
        self._std=(0.229, 0.224, 0.225)

        # Create Detector
        if (self.network_name.find('custom')>=0):
            classes=[str(i) for i in range(0,num_classes)]
            self.net = gcv.model_zoo.get_model(self.network_name[len('custom-'):], classes=classes, pretrained_base=False)
            self.net.reset_class(classes)
            self.net.load_parameters(self.prefix)
        else:
            self.net = gcv.model_zoo.get_model(self.network_name, pretrained=True)

        self.net.set_nms(nms_thresh=0.45, nms_topk=400)
        self.net.collect_params().reset_ctx(self.ctx)
        self.batch_data=None

    def detect(self, image, threshold):
        dets=[]
        num_detections=0
        # operate on list of images
        if (type(image)!=type(list())):
            image = [image]
        orig_image_size=image[0].shape[1]
        # set image_size to that specified on command line
        image_size=self.image_size
        #if (self.batch_data is None):
        #    self.batch_data=mx.ndarray.zeros((self.batch_size,3,image_size,image_size))

        # pack list of images into batch size as appropriate
        num_loops = int(np.ceil(len(image)/float(self.batch_size)))
        for i in range(0,num_loops):
            self.batch_data=mx.ndarray.zeros((self.batch_size,3,image_size,image_size))
            start_ind=i*self.batch_size
            stop_ind=min((i+1)*self.batch_size,len(image))
            for j in range(start_ind,stop_ind):
                image_np = image[j]
                data = mxnet.nd.array(image_np)
                if (image_size != orig_image_size):
                    data = mx.image.imresize(data, image_size, image_size)
                data = mx.nd.image.to_tensor(data)
                data = mx.nd.image.normalize(data, mean=self._mean, std=self._std)
                self.batch_data[j-start_ind,:,:,:]=data
            
            self.batch_data = self.batch_data.as_in_context(self.ctx)
            ids, scores, bboxes = [xx.asnumpy() for xx in self.net(self.batch_data)]

            # loop over batch
            for j in range(0,stop_ind-start_ind):
                detections=[]
                for det in zip(ids[j,:], scores[j,:], bboxes[j,:,:]):
                    cid = det[0]
                    if cid < 0:
                        continue
                    score = det[1]
                    if score < threshold:
                        continue
                    bbox=det[2]/image_size
                    detections.append([cid,score,bbox[0],bbox[1],bbox[2],bbox[3]])
                detections=np.asarray(detections)
                dets.append(detections)
                num_detections=num_detections+detections.shape[0]

        # detections are in form [[cls, prob, xmin, ymin, xmax, ymax]]
        return dets,num_detections

