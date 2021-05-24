#!/usr/bin/env python

import numpy as np
import cv2
import itertools


class SSDCropPattern():
    def __init__(self, zoom_enabled, level0_ncrops, level1_xcrops, level1_ycrops, level1_crop_size):
        # are we applying the sliding window/zoom crop pattern or simply passing one image to detector?
        self.zoom_enabled = zoom_enabled
        self.zoom_set_tmp = zoom_enabled
        self.zoom_changed = False
        # counter to help us determine if safe to change zoom parameter (the crop decoder must be matched to the encode crop pattern)
        self.encode_decode = 0

        # crop pattern
        self.level0_ncrops = level0_ncrops
        self.level1_xcrops = level1_xcrops
        self.level1_ycrops = level1_ycrops
        self.level1_crop_size = level1_crop_size

        # set data_shape to None, on first encoding, this gets set to the frame size
        self.data_shape = None
        self.COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255),
                       (255, 255, 255)]

    # callback to set zoom parameter internally in a safe manner
    def set_zoom(self, zoom_enabled):
        self.zoom_set_tmp = zoom_enabled
        self.zoom_changed = True

    # draw detections (must have been decoded if using any crop)
    def overlay_detections(self, frame, detarray):
        # draw detection box with one of 7 unique colors (modulo if number of classes is greater than 7)
        for det in detarray:
            pt1 = (int(det[2] * self.data_shape[1]), int(det[3] * self.data_shape[0]))
            pt2 = (int(det[4] * self.data_shape[1]), int(det[5] * self.data_shape[0]))
            cv2.rectangle(frame, pt1, pt2, self.COLORS[int(det[0]) % len(self.COLORS)], 2)
        return frame

    def iou(self, x, ys):
        """
        Taken from Joshua Zheng's (zhreshold) mxnet-ssd repo
        Calculate intersection-over-union overlap
        Params:
        ----------
        x : numpy.array
            single box [xmin, ymin ,xmax, ymax]
        ys : numpy.array
            multiple box [[xmin, ymin, xmax, ymax], [...], ]
        Returns:
        -----------
        numpy.array
            [iou1, iou2, ...], size == ys.shape[0]
        """
        ixmin = np.maximum(ys[:, 0], x[0])
        iymin = np.maximum(ys[:, 1], x[1])
        ixmax = np.minimum(ys[:, 2], x[2])
        iymax = np.minimum(ys[:, 3], x[3])
        iw = np.maximum(ixmax - ixmin, 0.)
        ih = np.maximum(iymax - iymin, 0.)
        inters = iw * ih
        uni = (x[2] - x[0]) * (x[3] - x[1]) + (ys[:, 2] - ys[:, 0]) * \
              (ys[:, 3] - ys[:, 1]) - inters
        ious = inters / uni
        ious[uni < 1e-12] = 0  # in case bad boxes
        return ious

    def prune_overlapping_boxes(self, list_of_all_detections):
        # takes in list containing the detections, for each crop location.
        # returns all_detections nparray of pruned detectections arrays
        # note, all_detections is not a list anymore after this function
        keep_inds = []
        # convert to sparse list of only detections made in crops
        all_detections = np.zeros((0, 6))
        for i in range(0, len(list_of_all_detections)):
            detections = list_of_all_detections[i]
            if len(detections) > 0:
                all_detections = np.vstack((all_detections, detections))

        # if detections, loop over boxes, find those with any non-negligible overlap, and prune all but the one with largest metric
        if (len(all_detections) > 0):
            areas = (all_detections[:, 4] - all_detections[:, 2]) * (all_detections[:, 5] - all_detections[:, 3])
            probs = all_detections[:, 1]
            metric = areas * probs * probs
            discard_inds = []
            for i in range(0, len(all_detections)):
                # get overlap between each box here and all other boxes
                ious = self.iou(1000.0 * all_detections[i, 2:], 1000.0 * all_detections[:, 2:])
                # check if the class matches for the overlapping boxes
                # if different class, set the iou to zero so we keep them both
                ious[np.where(all_detections[:, 0] != all_detections[i, 0])] = 0.0
                # take only those with IOU greater than 1%
                inds_to_prune_over = np.asarray(np.where(ious > 0.01))
                # find the "best" one and prune the others
                discard_inds.extend(
                    inds_to_prune_over[np.where(metric[inds_to_prune_over] != np.max(metric[inds_to_prune_over]))])
                # TODO identify mergeable overlapping boxes

            discard_inds = np.asarray(discard_inds).flatten()
            discard_inds = np.unique(discard_inds)
            keep_inds = np.asarray(np.setxor1d(range(0, len(all_detections)), discard_inds), dtype=np.int)

        if (len(keep_inds) > 0):
            all_detections = all_detections[keep_inds, :]

        return all_detections

    def decode_crops(self, all_detections):
        if (self.data_shape is None):
            print('Not initialized, can only decode if encoded at least once')
            return []

        # decode the crops back to original image
        pct_indices, _, _, _ = self.get_crop_location_pcts(report_overlaps=False)
        if (len(all_detections) != len(pct_indices)):
            print('WARNING, crop pattern (len=' + str(len(pct_indices)) + ') does not match detections (len=' + str(
                len(all_detections)) + ')')
        ydim, xdim = self.data_shape[0:2]
        for i in range(0, len(pct_indices)):
            xc, yc, w, h = pct_indices[i, :]
            x1 = max(xc - (w / 2), 0.0)
            y1 = max(yc - (h / 2), 0.0)
            # rebox the detections into the innerbox xc[i]-w[i]/2.0:xc[i]+w[i]/2.0
            for j in range(0, len(all_detections[i])):
                all_detections[i][j, 2] = all_detections[i][j, 2] * w + x1
                all_detections[i][j, 3] = all_detections[i][j, 3] * h + y1
                all_detections[i][j, 4] = all_detections[i][j, 4] * w + x1
                all_detections[i][j, 5] = all_detections[i][j, 5] * h + y1

        # prune overlapping boxes with same class id
        # print("antes: " + str(all_detections))
        all_detections=self.prune_overlapping_boxes(all_detections)
        # print("despues: "+str(all_detections))
        # all_detections_new = []
        # if len(all_detections[0]) > 0:
        #     all_detections_new.extend(all_detections[0])
        # if len(all_detections[1]) > 0:
        #     all_detections_new.extend(all_detections[1])

        # print("despues: " + str(all_detections_new))

        # decrement the encode_decode back to zero
        self.encode_decode = self.encode_decode - 1
        return all_detections

    # create lists containing the box centers and the box sizes in pcts = xc,yc,w,h
    def get_crop_location_pcts(self, report_overlaps=False, data_shape=(480, 640, 3)):
        if (report_overlaps):
            self.data_shape = data_shape
        ydim, xdim = self.data_shape[0:2]
        min_dim = np.min(self.data_shape[0:2])
        max_dim = np.max(self.data_shape[0:2])
        # First-level crop pattern
        xcrop_pct = float(min_dim) / xdim
        ycrop_pct = float(min_dim) / ydim
        xc0 = np.linspace(0.5 * xcrop_pct, 1.0 - 0.5 * xcrop_pct, self.level0_ncrops)
        yc0 = np.linspace(0.5 * ycrop_pct, 1.0 - 0.5 * ycrop_pct, self.level0_ncrops)
        w0 = np.asarray([xcrop_pct] * len(xc0))
        h0 = np.asarray([ycrop_pct] * len(yc0))
        # special pattern #1: if ncrops=0, set to center and full size
        if (self.level0_ncrops == 0):
            xc0 = np.asarray([0.5])
            yc0 = np.asarray([0.5])
            w0 = np.asarray([1.0])
            h0 = np.asarray([1.0])
        # special pattern #2: if ncrops=1, set to center and crop size
        if (self.level0_ncrops == 1):
            xc0 = np.asarray([0.5])
            yc0 = np.asarray([0.5])
            w0 = np.asarray([xcrop_pct])
            h0 = np.asarray([ycrop_pct])
        pct_indices0 = np.asarray([[a, b, c, d] for a, b, c, d in zip(xc0, yc0, w0, h0)])
        # determine the overlap percentages
        if (report_overlaps):
            if (self.level0_ncrops > 1):
                level0_overlap = 1.0 - (max(abs(xc0[0] - xc0[1]), abs(yc0[0] - yc0[1])) / min(xcrop_pct, ycrop_pct))
            else:
                level0_overlap = 0.0

        # Second-level crop pattern - if xcrops or ycrops are 0, then those lists will be zero length
        xcrop_pct = float(self.level1_crop_size) / xdim
        ycrop_pct = float(self.level1_crop_size) / ydim
        # WARNING: we assume the user wants to use this for improving distance detection - if level1_crop is 1 or less in either dimension, it won't make sense
        xc = np.linspace(0.5 * xcrop_pct, 1.0 - 0.5 * xcrop_pct, self.level1_xcrops)
        yc = np.linspace(0.5 * ycrop_pct, 1.0 - 0.5 * ycrop_pct, self.level1_ycrops)
        w = np.asarray([xcrop_pct] * len(xc))
        h = np.asarray([ycrop_pct] * len(yc))
        # these have to be replicated combinatorically and zipped so we have the total possibilities of xc,yc and same for w,h
        pct_indices1 = np.asarray([[i[0][0], i[0][1], i[1][0], i[1][1]] for i in
                                   zip(list(itertools.product(xc, yc, repeat=1)),
                                       list(itertools.product(w, h, repeat=1)))])
        #
        if (report_overlaps):
            if (self.level1_xcrops > 1 or self.level1_ycrops > 1):
                level1_xoverlap = 1.0 - (abs(xc[0] - xc[1]) / xcrop_pct)
                level1_yoverlap = 1.0 - (abs(yc[0] - yc[1]) / ycrop_pct)
            else:
                level1_xoverlap = 0.0
                level1_yoverlap = 0.0

        # stack the levels if zoom is set and non-zero length of crop lists
        if (self.zoom_enabled and len(xc) > 0 and len(yc) > 0):
            pct_indices0 = np.vstack((pct_indices0, pct_indices1))

        if (report_overlaps):
            return pct_indices0, level0_overlap, level1_xoverlap, level1_yoverlap

        return pct_indices0, None, None, None

    def encode_crops(self, frame):
        # check for Safe Zoom change-on-the-fly
        # only change the zoom value if we're NOT in between encodings and decodings, so check it here before we start encoding
        if (self.zoom_changed and self.encode_decode == 0):
            self.zoom_changed = False
            self.zoom_enabled = self.zoom_set_tmp
        # increment the encoded images count for safety in decoding same pattern we encoded with
        self.encode_decode = self.encode_decode + 1

        # on to the encoding of crops, get image dimensions
        self.data_shape = frame.shape
        ydim, xdim = self.data_shape[0:2]
        min_dim = np.min(self.data_shape[0:2])
        max_dim = np.max(self.data_shape[0:2])

        # make the list of crops
        framelist = []
        # If level0_ncrops==0, resize to square and return
        if (self.level0_ncrops == 0):
            framelist.append(cv2.resize(np.copy(frame), (min_dim, min_dim)))
            return framelist

        # otherwise, get the crop indices, loop over the crop indices and add a copy of each crop to framelist
        pct_indices, _, _, _ = self.get_crop_location_pcts(report_overlaps=False)
        ydim, xdim = self.data_shape[0:2]
        for i in range(0, len(pct_indices)):
            xc, yc, w, h = pct_indices[i, :]
            x1 = max(int((xc - (w / 2)) * xdim), 0)
            y1 = max(int((yc - (h / 2)) * ydim), 0)
            x2 = min(int((xc + (w / 2)) * xdim), xdim)
            y2 = min(int((yc + (h / 2)) * ydim), ydim)
            framelist.append(np.copy(frame[y1:y2, x1:x2, :]))

        return framelist

    def mask_detections(self, detections, mask, overlap):
        num_detections = len(detections)
        if (overlap <= 0.0 or mask is None):
            return detections, num_detections

        # if overlap is set, eliminate those with less than overlap percentage with mask==255 values
        discard_inds = []
        for i, det in enumerate(detections):
            box = det[2:]
            xmin = int(box[0] * mask.shape[1])
            ymin = int(box[1] * mask.shape[0])
            xmax = int(box[2] * mask.shape[1])
            ymax = int(box[3] * mask.shape[0])
            maskpct = 100.0 * np.mean(mask[ymin:ymax, xmin:xmax]) / 255.0
            if (maskpct < overlap):
                # print 'mask below overlap threshold at pct='+str(maskpct)
                discard_inds.append(i)

        discard_inds = np.asarray(discard_inds).flatten()
        discard_inds = np.unique(discard_inds)
        keep_inds = np.asarray(np.setxor1d(range(0, len(detections)), discard_inds), dtype=np.int)
        if (len(keep_inds) > 0):
            detections = detections[keep_inds]
        else:
            detections = []

        return detections, len(detections)


def convert_frame_to_jpeg_string(frame):
    return np.array(cv2.imencode('.jpg', frame[:, :, [2, 1, 0]])[1]).tostring()


def write_image_detection(filename, image):
    cv2.imwrite(filename, image)
