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
from geometry_msgs.msg import PoseStamped, Point, Vector3, PoseWithCovariance
from sensor_msgs.msg import Image, CompressedImage
from vimantic.msg import SemanticObject, SemanticObjectArray
from vision_msgs.msg import ObjectHypothesis


class ViManticNode(object):
    def __init__(self):
        rospy.logwarn("Initializing")
        # ROS Parameters
        self.image_topic = self.load_param('~topic_virtualCameraRGBD', "ViMantic/virtualCameraRGBD")
        self.semantic_topic = self.load_param('~topic_result', 'ViMantic/SemanticObjects')
        self.cnn_topic = self.load_param('~topic_cnn')
        self.image_toCNN = self.load_param('~topic_republic', 'ViMantic/ToCNN')
        self.debug = self.load_param('~debug', False)

        # Camera calibration
        self.cx = 320
        self.cy = 240
        self.fx = 457.1429
        self.fy = 470.5882

        # General Variables
        self._bridge = CvBridge()
        self._tfBuffer = tf2_ros.Buffer()
        self._image_counter = 0
        self._last_msg = None
        self._waiting_cnn = False
        self._tries = 0

        # Publisher
        self._pub_result = rospy.Publisher(self.semantic_topic, SemanticObjectArray, queue_size=10)
        self._pub_repub = rospy.Publisher(self.image_toCNN, Image, queue_size=1)

        if self.debug:
            self._pub_pose = rospy.Publisher('ViMantic/detectedObject', PoseStamped, queue_size=1)

        # Subscribers
        rospy.Subscriber(self.cnn_topic, detectron2_ros.msg.Result, self.callback_new_detection)
        rospy.Subscriber(self.image_topic, CompressedImage, self.callbackVirtualImage, queue_size=10)

        tf2_ros.TransformListener(self._tfBuffer)
        self.start_time = 0
        rospy.logwarn("Initialized")

    def run(self):

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            # Republish last img
            if self._waiting_cnn and self._tries > 200:
                self._pub_repub.publish(self._bridge.cv2_to_imgmsg(self._last_msg[1], 'rgb8'))
                self._waiting_cnn = False
                rospy.logwarn("[ViMantic] CNN does not respond, trying again.")
            else:
                self._tries += 1

            rate.sleep()

    def callbackVirtualImage(self, img_msg):

        if not self._waiting_cnn:
            transform = self._tfBuffer.lookup_transform("map",
                                                        img_msg.header.frame_id,  # source frame
                                                        rospy.Time(0))  # get the tf at first available time

            np_arr = np.fromstring(img_msg.data, np.uint8)
            im = cv2.imdecode(np_arr, -1)
            img_rgb = cv2.cvtColor(im[:, :, :3], cv2.COLOR_RGB2BGR)

            img_depth = np.divide(im[:, :, 3], 255.0)

            self._last_msg = [img_msg.header, img_rgb, img_depth, transform]
            self._pub_repub.publish(self._bridge.cv2_to_imgmsg(self._last_msg[1], 'rgb8'))
            self._tries = 0
            self._waiting_cnn = True
            if self.start_time == 0:
                self.start_time = time.time()

    def callback_new_detection(self, result_cnn):
        if self._waiting_cnn:
            self._image_counter = self._image_counter + 1
            if (self._image_counter % 11) == 10:
                rospy.loginfo("Images detected per second=%.2f",
                              float(self._image_counter) / (time.time() - self.start_time))

            if len(result_cnn.class_names) > 0:
                data_header = self._last_msg[0]
                img_depth = self._last_msg[2]
                data_transform = self._last_msg[3]

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

                for i in range(len(result_cnn.class_names)):

                    semanticObject = SemanticObject()
                    det = ObjectHypothesis()
                    det.id = result_cnn.class_names[i]
                    det.score = result_cnn.scores[i]
                    semanticObject.scores.append(det)


                    try:
                        mask = (self._bridge.imgmsg_to_cv2(result_cnn.masks[i]) == 255)
                    except CvBridgeError as e:
                        print(e)
                        continue

                    if x.shape != mask.shape:
                        return

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
                        return

                    scale_x = x_.max() - x_.min()
                    scale_y = y_.max() - y_.min()
                    scale_z = np.std(z_)

                    # semanticObject.size = Vector3(scale_z, scale_x , scale_y)
                    semanticObject.size = Vector3(scale_x, scale_z, scale_y)

                    # Calculate the center
                    x_center = int(result_cnn.boxes[i].x_offset + result_cnn.boxes[i].width / 2)
                    y_center = int(result_cnn.boxes[i].y_offset + result_cnn.boxes[i].height / 2)
                    z_center = scale_z / 2 + z_.min()

                    # Transformed the center of the object to the map reference system
                    p1 = PoseStamped()
                    p1.header = data_header
                    p1.pose.position = Point(z_center, x[y_center, x_center], y[y_center, x_center])
                    p1.pose.orientation.w = 1.0  # Neutral orientation
                    ans = tf2_geometry_msgs.do_transform_pose(p1, data_transform)

                    semanticObject.pose = PoseWithCovariance()
                    semanticObject.pose.pose = ans.pose

                    self._pub_pose.publish(ans)

                    result.semanticObjects.append(semanticObject)
                    objstring = objstring + ' ' + det.id + ', p=%.2f.' % (
                        det.score)

                self._pub_result.publish(result)
                rospy.loginfo(objstring)

            self._waiting_cnn = False

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
