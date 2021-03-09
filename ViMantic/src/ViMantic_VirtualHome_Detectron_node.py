#! /usr/bin/env python
import sys
import time

import cv2
import detectron2_ros.msg
import numpy as np
import rospy
import tf2_geometry_msgs
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, Point, Vector3
from numpy import savetxt
from sensor_msgs.msg import Image, CompressedImage
from vimantic.msg import SemanticObject, SemanticObjectArray


class ViManticNode(object):
    def __init__(self):
        rospy.logwarn("Initializing")
        # ROS Parameters
        self.image_topic = self.load_param('~topic_virtualCameraRGBD', "ViMantic/virtualCameraRGBD")
        self.semantic_topic = self.load_param('~topic_result', 'ViMantic/SemanticObjects')
        self.cnn_topic = self.load_param('~topic_cnn')
        self.image_toCNN = self.load_param('~topic_republic', 'ViMantic/ToCNN')
        self.debug = self.load_param('~debug', False)
        self.enableTimeCapture = self.load_param('~record_times', False)

        # Camera calibration
        self.cx = 320
        self.cy = 240
        self.fx = 457.1429
        self.fy = 470.5882

        # General Variables
        self.bridge = CvBridge()
        self.tfBuffer = tf2_ros.Buffer()
        self._image_counter = 0
        self.last_package = None
        self.cnn_msg = None
        self.waiting_cnn = False

        # Time variables
        self.time_cnn = 0
        self.time_objectInfoPacket = 0
        self.list_time_cnn = []
        self.list_time_objectInfoPacket = []

        # Publisher
        self._pub_result = rospy.Publisher(self.semantic_topic, SemanticObjectArray, queue_size=10)
        self._pub_repub = rospy.Publisher(self.image_toCNN, Image, queue_size=1)

        if (self.debug):
            self._pub_pose = rospy.Publisher('ViMantic/detectedObject', PoseStamped, queue_size=1)

        # Subscribers
        rospy.Subscriber(self.cnn_topic, detectron2_ros.msg.Result, self.callback_new_detection)
        rospy.Subscriber(self.image_topic, CompressedImage, self.callbackVirtualImage, queue_size=10)

        tf2_ros.TransformListener(self.tfBuffer)

        self.start_time = time.time()
        rospy.logwarn("Initialized")

    def run(self):

        rate = rospy.Rate(100)

        while not rospy.is_shutdown():
            # Republish last img
            if self.last_package is not None:
                self._pub_repub.publish(self.bridge.cv2_to_imgmsg(self.last_package[2], 'rgb8'))

            if self.cnn_msg is not None:
                self._image_counter = self._image_counter + 1
                if (self._image_counter % 11) == 10:
                    rospy.loginfo("Images detected per second=%.2f",
                                  float(self._image_counter) / (time.time() - self.start_time))

                # The detected objects are processed
                if len(self.cnn_msg.class_names) > 0:

                    if self.enableTimeCapture:
                        self.time_objectInfoPacket = rospy.get_rostime()

                    img_depth = self.last_package[3]
                    data_header = self.last_package[0]
                    data_transform = self.last_package[1]

                    # Transform the value of each px to m by acquiring a cloud of points
                    img_depth = img_depth * 15

                    rows, cols = img_depth.shape
                    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)

                    z = img_depth
                    x = ((self.cx - c) * z / self.fx)
                    y = ((self.cy - r) * z / self.fy)

                    # Cut out every object from the point cloud and build the result.
                    result = SemanticObjectArray()

                    result.header = data_header
                    result.header.frame_id = "/map"
                    objstring = 'Detected:'

                    for i in range(len(self.cnn_msg.class_names)):

                        semanticObject = SemanticObject()

                        semanticObject.object.score = self.cnn_msg.scores[i]
                        semanticObject.objectType = self.cnn_msg.class_names[i]

                        try:
                            mask = (self.bridge.imgmsg_to_cv2(self.cnn_msg.masks[i]) == 255)
                        except CvBridgeError as e:
                            print(e)
                            continue

                        if x.shape != mask.shape:
                            print(x.shape)
                            print(mask.shape)
                        else:
                            x_ = x[mask]
                            y_ = y[mask]
                            z_ = z[mask]

                            # Bandpass filter with Z data
                            top_margin = (z_.max() - z_.min()) * 0.9 + z_.min()
                            bottom_margin = (z_.max() - z_.min()) * 0.1 + z_.min()

                            mask2 = np.logical_and(z_ > bottom_margin, z_ < top_margin)
                            # mask2 = z_ < top_margin

                            x_ = x_[mask2]
                            y_ = y_[mask2]
                            z_ = z_[mask2]

                            if len(x_) == 0:
                                continue

                            scale_x = x_.max() - x_.min()
                            scale_y = y_.max() - y_.min()
                            scale_z = np.std(z_)

                            semanticObject.size = Vector3(scale_x, scale_z, scale_y)

                            # Calculate the center
                            x_center = int(self.cnn_msg.boxes[i].x_offset + self.cnn_msg.boxes[i].width / 2)
                            y_center = int(self.cnn_msg.boxes[i].y_offset + self.cnn_msg.boxes[i].height / 2)
                            z_center = scale_z / 2 + z_.min()

                            # Transformed the center of the object to the map reference system
                            p1 = PoseStamped()
                            p1.header = data_header
                            p1.pose.position = Point(z_center, x[y_center, x_center], y[y_center, x_center])
                            p1.pose.orientation.w = 1.0  # Neutral orientation
                            ans = tf2_geometry_msgs.do_transform_pose(p1, data_transform)
                            semanticObject.object.pose.pose = ans.pose

                            if self.debug:
                                self._pub_pose.publish(ans)

                            result.semanticObjects.append(semanticObject)
                            objstring = objstring + ' ' + semanticObject.objectType + ', p=%.2f.' % (
                                semanticObject.object.score)

                    self._pub_result.publish(result)
                    rospy.loginfo(objstring)

                    if self.enableTimeCapture:
                        self.list_time_objectInfoPacket.append(
                            (rospy.get_rostime() - self.time_objectInfoPacket).nsecs / 1000000)

                self.cnn_msg = None
                self.waiting_cnn = False

            rate.sleep()

        if self.enableTimeCapture:
            savetxt('~/ViMantic/time_cnn.csv', self.list_time_cnn, delimiter=',')
            savetxt('~/ViMantic/time_objectInfoPacket.csv', self.list_time_objectInfoPacket, delimiter=',')

    def callbackVirtualImage(self, img_msg):

        if not self.waiting_cnn:
            transform = self.tfBuffer.lookup_transform("map",
                                                       img_msg.header.frame_id,  # source frame
                                                       rospy.Time(0))  # get the tf at first available time

            np_arr = np.fromstring(img_msg.data, np.uint8)
            im = cv2.imdecode(np_arr, -1)
            img_rgb = cv2.cvtColor(im[:, :, :3], cv2.COLOR_RGB2BGR)

            img_depth = np.divide(im[:, :, 3], 255.0)

            self.last_package = [img_msg.header, transform, img_rgb, img_depth]
            self.waiting_cnn = True
            if self.enableTimeCapture:
                self.time_cnn = rospy.get_rostime()

    def callback_new_detection(self, result_cnn):
        if self.waiting_cnn and self.cnn_msg is None:
            self.cnn_msg = result_cnn
            # CNN Time
            if self.enableTimeCapture:
                self.list_time_cnn.append((rospy.get_rostime() - self.time_cnn).nsecs / 1000000)

    @staticmethod
    def load_param(param, default=None):
        new_param = rospy.get_param(param, default)
        rospy.loginfo("[ViMantic] %s: %s", param, new_param)
        return new_param


def main(argv):
    rospy.init_node('ViMantic')
    node = ViManticNode()
    node.run()


if __name__ == '__main__':
    main(sys.argv)
