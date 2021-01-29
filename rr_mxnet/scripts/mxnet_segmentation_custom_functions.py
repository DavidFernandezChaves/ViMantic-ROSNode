#!/usr/bin/env python

import numpy as np
import cv2
import itertools

class SegmentationFunctions():
    def __init__(self, mask_values=[12]):
        self.mask_values=mask_values

    def overlay_mask(self, image, mask, mask_color=[255,0,0], alpha=0.5):
        orig_image_size=image.shape
        # reshape image for indexing on pixels in 1D
        image=image.reshape((-1,3))
        mask=mask.reshape((-1,))

        inds = np.where(mask==255)[0]
        # set overlay to RED
        color=np.zeros_like(image)
        color[:,0]=mask_color[0]
        color[:,1]=mask_color[1]
        color[:,2]=mask_color[2]
        if len(inds)>0:
            image[inds,:]=cv2.addWeighted(image[inds,:],1-alpha,color[inds,:],alpha,0)

        # reshape back to original
        image=image.reshape((orig_image_size[0],orig_image_size[1],3))
        return image

    def segmentation_to_mask(self, segmentation):
        orig_image_size=segmentation.shape
        # reshape seg mask for indexing on pixels in 1D
        segmentation=segmentation.reshape((-1,))
        mask=np.zeros_like(segmentation)
        # AND the absolute value mask values together
        for cls in self.mask_values:
            inds = np.where(segmentation==abs(cls))[0]
            mask[inds]=255

        # handle negative values, use last one for detecting if negative
        if (int(abs(cls))==int(-1.0*cls)):
            inds = np.where(mask==0)[0]
            # invert the mask
            mask=0*mask
            mask[inds]=255

        # reshape back to original
        mask=mask.reshape((orig_image_size[0],orig_image_size[1]))
        return mask.astype(np.uint8)

    '''
    def average_frames(self, mask, accumulated_mask):
        # average frames if specified (not enabled at this time)
        if (self.run_once):
            self.frame_counter=self.frame_counter+1
            if (self.frame_counter==self.avg_segmentation_frames):
                self.run_once = False
    '''

