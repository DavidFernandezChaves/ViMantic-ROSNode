#! /usr/bin/env python
import sys
import time

import cv2
import detectron2_ros.msg
import message_filters
import numpy as np
import rospy
import tf
import tf2_geometry_msgs
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point32, PoseStamped, Point, Vector3, Quaternion, PoseWithCovariance
from numpy import savetxt
from vimantic.msg import SemanticObject, SemanticObjectArray
from sensor_msgs.msg import Image


class SemanticMappingNode(object):
    def __init__(self):
        rospy.logwarn("Initializing")

        # ROS Parameters
        self.image_rgb_topic = self.load_param('~topic_intensity')
        self.image_depth_topic = self.load_param('~topic_depth')
        self.semantic_topic = self.load_param('~topic_result', 'ViMantic/SemanticObjects')
        self.cnn_topic = self.load_param('~topic_cnn')
        self.image_toCNN = self.load_param('~topic_republic', 'ViMantic/ToCNN')
        self.input_angle = self.load_param('~input_angle', 0)
        self.debug = self.load_param('~debug', False)

        # Camera calibration
        self._cx = 314.649173 / 2
        self._cy = 240.160459 / 2
        self._fx = 572.882768
        self._fy = 542.739980

        # General Variables
        self._bridge = CvBridge()
        self._tfBuffer = tf2_ros.Buffer()
        self._image_counter = 0
        self._last_img_rgb = None
        self._last_msg = None
        self._cnn_msg = None
        self._waiting_cnn = False

        # Time variables
        self._enableTimeCapture = False
        self._time_cnn = 0
        self._time_objectInfoPacket = 0
        self._list_time_cnn = []
        self._list_time_objectInfoPacket = []

        # Publisher
        self._pub_result = rospy.Publisher(self.semantic_topic, SemanticObjectArray, queue_size=10)
        self._pub_repub = rospy.Publisher(self.image_toCNN, Image, queue_size=1)

        if (self.debug):
            self._pub_pose = rospy.Publisher('ViMantic/detectedObject', PoseStamped, queue_size=1)

        # Subscribers
        rospy.Subscriber(self.cnn_topic, detectron2_ros.msg.Result, self.callback_cnn)
        sub_rgb_image = message_filters.Subscriber(self.image_rgb_topic, Image)
        sub_depth_image = message_filters.Subscriber(self.image_depth_topic, Image)

        message_filter = message_filters.ApproximateTimeSynchronizer([sub_depth_image, sub_rgb_image], 10, 0.3)
        message_filter.registerCallback(self.callback_img)

        tf2_ros.TransformListener(self._tfBuffer)
        self.start_time = time.time()
        rospy.logwarn("Initialized")

    def run(self):

        rate = rospy.Rate(100)

        while not rospy.is_shutdown():
            # Republish last img
            if self._last_img_rgb is not None:
                self._pub_repub.publish(self._bridge.cv2_to_imgmsg(self._last_img_rgb, 'rgb8'))

            if self._cnn_msg is not None:
                self._image_counter = self._image_counter + 1
                if (self._image_counter % 11) == 10:
                    rospy.loginfo("Images detected per second=%.2f",
                                  float(self._image_counter) / (time.time() - self.start_time))

                # The detected objects are processed
                if len(self._cnn_msg.class_names) > 0:

                    if self._enableTimeCapture:
                        self._time_objectInfoPacket = rospy.get_rostime()

                    img_depth = self._last_msg[0]
                    data_header = self._last_msg[1]
                    data_transform = self._last_msg[2]

                    # Transform the value of each px to m by acquiring a cloud of points
                    img_depth = img_depth / 6553.5
                    img_depth = self.rotate_image(img_depth, self.input_angle)

                    rows, cols = img_depth.shape
                    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)

                    z = img_depth
                    x = ((self._cx - c) * z / self._fx)
                    y = ((self._cy - r) * z / self._fy)

                    # Cut out every object from the point cloud and build the result.
                    result = SemanticObjectArray()

                    result.header = data_header
                    result.header.frame_id = "/map"
                    objstring = 'Detected:'
                    for i in range(len(self._cnn_msg.class_names)):

                        semanticObject = SemanticObject()

                        semanticObject.object.score = self._cnn_msg.scores[i]
                        semanticObject.objectType = self._cnn_msg.class_names[i]

                        try:
                            mask = (self._bridge.imgmsg_to_cv2(self._cnn_msg.masks[i]) == 255)
                        except CvBridgeError as e:
                            print(e)
                            continue

                        x_ = x[mask]
                        y_ = y[mask]
                        z_ = z[mask]

                        # Bandpass filter with Z data
                        top_margin = (z_.max() - z_.min()) * 0.9 + z_.min()
                        bottom_margin = (z_.max() - z_.min()) * 0.1 + z_.min()

                        mask2 = np.logical_and(z_ > bottom_margin, z_ < top_margin)

                        x_ = x_[mask2]
                        y_ = y_[mask2]
                        z_ = z_[mask2]

                        if len(x_) == 0:
                            continue

                        scale_x = x_.max() - x_.min()
                        scale_y = y_.max() - y_.min()
                        scale_z = np.std(z_)

                        semanticObject.size = Vector3(scale_x, scale_y, scale_z)

                        # Calculate the center px
                        x_center = int(self._cnn_msg.boxes[i].x_offset + self._cnn_msg.boxes[i].width / 2)
                        y_center = int(self._cnn_msg.boxes[i].y_offset + self._cnn_msg.boxes[i].height / 2)
                        # And the depth of the center
                        z_center = -(float(scale_z / 2) + np.average(z_))

                        # Transformed the center of the object to the map reference system
                        p1 = PoseStamped()
                        p1.header = data_header

                        p1.pose.position = Point(-x[y_center, x_center], y[y_center, x_center], z_center)
                        p1.pose.orientation.w = 1.0  # Neutral orientation
                        ans = tf2_geometry_msgs.do_transform_pose(p1, data_transform)
                        semanticObject.object.pose = PoseWithCovariance()
                        semanticObject.object.pose.pose = ans.pose

                        if self.debug:
                            self._pub_pose.publish(ans)

                        result.semanticObjects.append(semanticObject)
                        objstring = objstring + ' ' + semanticObject.objectType + ', p=%.2f.' % (
                            semanticObject.object.score)

                    self._pub_result.publish(result)
                    rospy.loginfo(objstring)

                    if self._enableTimeCapture:
                        self._list_time_objectInfoPacket.append(
                            (rospy.get_rostime() - self._time_objectInfoPacket).nsecs / 1000000)

                self._cnn_msg = None
                self._waiting_cnn = False

            rate.sleep()

        if self._enableTimeCapture:
            savetxt('~/ViMantic/time_cnn.csv', self._list_time_cnn, delimiter=',')
            savetxt('~/ViMantic/time_objectInfoPacket.csv', self._list_time_objectInfoPacket, delimiter=',')

    def callback_img(self, depth_msg, rgb_msg):

        if not self._waiting_cnn:
            try:
                img_rgb = self._bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
                img_depth = self._bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            except CvBridgeError as e:
                print(e)

            self._last_img_rgb = self.rotate_image(img_rgb, self.input_angle)

            transform = self._tfBuffer.lookup_transform("map",
                                                        rgb_msg.header.frame_id,  # source frame
                                                        rospy.Time(0),  # get the tf at first available time
                                                        rospy.Duration(5))

            # Robot@Home fixe
            # rotation = (transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z,
            #            transform.transform.rotation.w)
            # rot = tf.transformations.euler_from_quaternion(rotation)
            # newrot = tf.transformations.quaternion_from_euler(rot[0], rot[1], rot[2] - 90)
            # transform.transform.rotation = Quaternion(newrot[0], newrot[1], newrot[2], newrot[3])

            self._last_msg = [img_depth, depth_msg.header, transform]
            self._waiting_cnn = True
            if self._enableTimeCapture:
                self._time_cnn = rospy.get_rostime()

    def callback_cnn(self, result_cnn):
        if self._waiting_cnn and self._cnn_msg is None:
            self._cnn_msg = result_cnn
            # CNN Time
            if self._enableTimeCapture:
                self._list_time_cnn.append((rospy.get_rostime() - self._time_cnn).nsecs / 1000000)

    @staticmethod
    def load_param(param, default=None):
        new_param = rospy.get_param(param, default)
        rospy.loginfo("[ViMantic] %s: %s", param, new_param)
        return new_param

    @staticmethod
    def rotate_image(img, angle):
        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result


def main(argv):
    rospy.init_node('semantic_mapping')
    node = SemanticMappingNode()
    node.run()


if __name__ == '__main__':
    main(sys.argv)
