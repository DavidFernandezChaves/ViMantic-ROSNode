#!/usr/bin/env python
import cv2 as cv
import rospy
import time
import os
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Bool, Int32
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge, CvBridgeError
from mxnet_ssd import MxNetSSDClassifier
from mxnet_ssd_custom_functions import SSDCropPattern, convert_frame_to_jpeg_string, write_image_detection
from datetime import datetime


class RosMxNetSSD:
    def __init__(self):
        rospy.logwarn("Initializing")

        # ROS Parameters
        rospy.loginfo("[MXNET] Loading ROS Parameters")
        self.image_topic = self.load_param('~image_topic', '/usb_cam/image_raw')
        self.detections_topic = self.load_param('~detections_topic', '~detections')
        self.publish_detection_images = self.load_param('~publish_detection_images', False)
        self.image_detections_topic = self.load_param('~image_detections_topic', '~image')
        self.timer = self.load_param('~throttle_timer', 5)
        self.latency_threshold_time = self.load_param('~latency_threshold', 2)
        self.threshold = self.load_param('~threshold', 0.5)
        self.start_enabled = self.load_param('~start_enabled', False)
        self.zoom_enabled = self.load_param('~start_zoom_enabled', False)

        # crop pattern
        self.level0_ncrops = self.load_param('~level0_ncrops', 2)
        self.level1_xcrops = self.load_param('~level1_xcrops', 4)
        self.level1_ycrops = self.load_param('~level1_ycrops', 2)
        self.level1_crop_size = self.load_param('~level1_crop_size', 380)

        # location of mxnet model and name, GPU and number of classes
        self.classes = self.load_param('~classes',
                                       'aeroplane, bicycle, bird, boat, bottle, bus,car, cat, chair, cow, diningtable, dog, horse, motorbike,person, pottedplant, sheep, sofa, train, tvmonitor')
        self.enable_gpu = self.load_param('~enable_gpu', True)
        # recommendation to use self.level0_ncrops in most cases for batch_size
        self.batch_size = self.load_param('~batch_size', 1)
        self.network = self.load_param('~network', 'ssd_512_resnet50_v1_voc')
        self.model_filename = self.load_param('~model_filename', '')
        self.model_directory = self.load_param('~model_directory', os.environ['HOME'] + '/mxnet_ssd/')
        # if setting a custom model, need to specify the network will be custom (see below), 
        # specify the model filename and put model in specified directory
        # self.network = 'custom-ssd_512_resnet50_v1_custom'
        # self.model_filename = 'ssd_512_resnet50_v1_custom.params'

        # save detections output and location
        self.save_detections = self.load_param('~save_detections', False)
        self.save_directory = self.load_param('~save_directory', '/tmp')
        # use some unique value, e.g. hostname or other
        self.save_prefix = self.load_param('~save_prefix', 'detector')

        # mask detections
        self.mask_topic = self.load_param('~mask_topic', '/rr_mxnet_segmentation/segmentation_mask')
        self.mask_overlap_param = self.load_param('~mask_overlap_param', 0)

        # Class Variables
        self.detection_seq = 0
        self.camera_frame = "camera_frame"
        self.classes = [c.strip(' ') for c in self.classes.strip('\n').split(',')]
        self.num_classes = len(self.classes)
        self.last_detection_time = 0
        self.reported_overlaps = False
        self.data_shape = None
        self.image_counter = 0
        self.mask = None


        # SSD classes for handling data/detections
        self.classifier = MxNetSSDClassifier(self.model_directory, self.model_filename, self.network, self.batch_size,
                                             self.enable_gpu, self.num_classes)
        self.imageprocessor = SSDCropPattern(self.zoom_enabled, self.level0_ncrops, self.level1_xcrops,
                                             self.level1_ycrops, self.level1_crop_size)

        # ROS Subscribers and variables used in their callbacks
        self.start_time = time.time()
        self.bridge = CvBridge()
        self.sub_image = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        self.sub_enable = rospy.Subscriber('~enable', Bool, self.enable_cb, queue_size=1)
        self.sub_zoom = rospy.Subscriber('~zoom', Bool, self.zoom_cb, queue_size=1)
        self.sub_mask = rospy.Subscriber(self.mask_topic, Image, self.mask_cb, queue_size=1)
        self.sub_overlap = rospy.Subscriber('~mask_overlap', Int32, self.overlap_cb, queue_size=1)

        # ROS Publishers
        self.pub_detections = rospy.Publisher(self.detections_topic, Detection2DArray, queue_size=10)
        if (self.publish_detection_images):
            # publish uncompressed image
            self.pub_img_detections = rospy.Publisher(self.image_detections_topic, Image, queue_size=1)
            # compressed image topic must end in /compressed
            self.pub_img_compressed_detections = rospy.Publisher(self.image_detections_topic + "/compressed",
                                                                 CompressedImage, queue_size=1)

        rospy.loginfo("[MxNet Initialized with model %s]", self.model_filename)

    def load_param(self, param, default=None):
        new_param = rospy.get_param(param, default)
        rospy.loginfo("[MxNet] %s: %s", param, new_param)
        return new_param

    def mask_cb(self, image):
        rospy.loginfo("Obtaining mask")
        try:
            cv2_img = cv2_img = self.convert_to_cv_image(image)
            self.mask = np.asarray(cv2_img).copy()
            rospy.loginfo("Received mask")
        except CvBridgeError as e:
            rospy.logerr(e)

    def overlap_cb(self, msg):
        overlap = msg.data
        rospy.loginfo("Setting mask overlap required to %d", overlap)
        if (overlap >= 0 and overlap <= 100):
            self.mask_overlap_param = overlap

    def enable_cb(self, msg):
        self.start_enabled = msg.data
        rospy.loginfo("MxNet enable_cb: " + str(self.start_enabled))

    def zoom_cb(self, msg):
        # set_zoom is safe, it doesn't take effect until the count of encoded and decoded are equal
        # allows the zoom setting to be changed on the fly
        self.zoom_enabled = msg.data
        self.imageprocessor.set_zoom(self.zoom_enabled)
        rospy.loginfo("MxNet zoom_cb: " + str(self.zoom_enabled))

    def encode_detection_msg(self, detections):
        detections_msg = Detection2DArray()
        if len(detections) > 0:
            i = 0
            detstring = 'Detected:'
            for det in detections:
                detection = Detection2D()
                detection.header.seq = self.detection_seq
                detection.header.stamp = rospy.Time.now()
                detection.header.frame_id = self.camera_frame
                result = [ObjectHypothesisWithPose()]
                result[0].id = self.classes[int(det[0])]
                result[0].score = det[1]
                detection.results = result
                detection.bbox.size_x = int(min(det[4],1) * self.data_shape[1]) - int(min(det[2],1) * self.data_shape[1])
                detection.bbox.size_y = int(min(det[5],1) * self.data_shape[0]) - int(min(det[3],1) * self.data_shape[0])
                detection.bbox.center.x = int(min(det[2],1)*self.data_shape[1] + detection.bbox.size_x /2)
                detection.bbox.center.y = int(min(det[3],1)*self.data_shape[0] + detection.bbox.size_y /2)
                detections_msg.detections.append(detection)
                detstring = detstring + ' ' + self.classes[int(det[0])] + ', p=%.2f.' % (det[1])
            rospy.logwarn(detstring)
        self.detection_seq += 1
        return detections_msg

    def convert_to_cv_image(self, image_msg):

        if image_msg is None:
            return None

        self._width = image_msg.width
        self._height = image_msg.height
        channels = int(len(image_msg.data) / (self._width * self._height))

        encoding = None
        if image_msg.encoding.lower() in ['rgb8', 'bgr8']:
            encoding = np.uint8
        elif image_msg.encoding.lower() == 'mono8':
            encoding = np.uint8
        elif image_msg.encoding.lower() == '32fc1':
            encoding = np.float32
            channels = 1

        cv_img = np.ndarray(shape=(image_msg.height, image_msg.width, channels),
                            dtype=encoding, buffer=image_msg.data)

        if image_msg.encoding.lower() == 'mono8':
            cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2GRAY)
        else:
            cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2BGR)

        return cv_img

    def report_overlaps(self):
        pct_indices, level0_overlap, level1_xoverlap, level1_yoverlap = self.imageprocessor.get_crop_location_pcts(
            report_overlaps=True, data_shape=self.data_shape)
        rospy.loginfo("Getting Overlap...")
        rospy.logwarn(
            "For Input Image Shape=%d,%d,%d, overlap first-level: %d%%, second-level zoom overlap: %d%% x %d%%",
            self.data_shape[0], self.data_shape[1], self.data_shape[2], int(100 * level0_overlap),
            int(100 * level1_xoverlap), int(100 * level1_yoverlap))
        rospy.loginfo("Overlap Reporting Done.")
        self.start_time = time.time()

    def image_cb(self, image):
        if (not self.reported_overlaps):
            cv2_img = self.convert_to_cv_image(image)

            # get and report the overlap percentages
            self.data_shape = cv2_img.shape
            self.report_overlaps()
            self.reported_overlaps = True

        # check latency and return if more than latency threshold out of date
        # current_time = rospy.get_rostime().secs + float(int(rospy.get_rostime().nsecs/1000000.0))/1000.0
        # image_time = image.header.stamp.secs + float(int(image.header.stamp.nsecs/1000000.0))/1000.0
        # if (current_time-image_time>self.latency_threshold_time):
        #     return

        if self.start_enabled:
            current_time = rospy.get_rostime().secs
            if self.last_detection_time + self.timer <= current_time:
                self.last_detection_time = current_time
                try:
                    cv2_img = self.convert_to_cv_image(image)
                    frame = np.asarray(cv2_img).copy()
                    self.image_counter = self.image_counter + 1
                    if (self.image_counter % 11) == 10:
                        rospy.loginfo("Images detected per second=%.2f",
                                      float(self.image_counter) / (time.time() - self.start_time))

                    # get list of crops, encode image into the specified crop pattern if zoom is enabled
                    framelist = self.imageprocessor.encode_crops(frame)

                    # pass all frame crops to classifier, returns a list of detections, or a list of zeros if nothing detected in a crop
                    # e.g. every crop gets a detection array added to the list
                    list_of_crop_detections, num_detections = self.classifier.detect(framelist, self.threshold)

                    # decode the detections list for the encoded crop pattern into original image locations
                    decoded_image_detections = self.imageprocessor.decode_crops(list_of_crop_detections)

                    # apply mask (if present) at the desired required overlap percentage
                    masked_detections, num_detections = self.imageprocessor.mask_detections(decoded_image_detections,
                                                                                            self.mask,
                                                                                            self.mask_overlap_param)

                    # if there are no detections, continue
                    # if num_detections==0:
                    #     return

                    # package up the list of detections as a message and publish
                    detections_msg = self.encode_detection_msg(masked_detections)
                    self.pub_detections.publish(detections_msg)

                    # if specified, publish images with bounding boxes if detections present
                    if (self.publish_detection_images):
                        # overlay detections on the frame
                        frame = self.imageprocessor.overlay_detections(frame, decoded_image_detections)
                        try:
                            # send uncompressed image
                            self.pub_img_detections.publish(self.bridge.cv2_to_imgmsg(frame))
                            # send compressed image
                            msg = CompressedImage()
                            msg.header.stamp = rospy.Time.now()
                            msg.format = "jpeg"
                            msg.data = convert_frame_to_jpeg_string(frame)
                            self.pub_img_compressed_detections.publish(msg)
                        except CvBridgeError as e:
                            rospy.logerr(e)

                    # save image file with some unique prefix and timestamp
                    if (self.save_detections):
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        write_image_detection(
                            self.save_directory + '/' + self.save_prefix + '_' + timestamp + '_mxdet_%05d.jpg' % (
                                self.detection_seq), frame[:, :, [2, 1, 0]])

                except CvBridgeError as e:
                    rospy.logerr(e)


if __name__ == '__main__':
    rospy.init_node("rr_mxnet_ssd", anonymous=False, log_level=rospy.INFO)
    ros_mxnet_ssd = RosMxNetSSD()
    rospy.spin()
