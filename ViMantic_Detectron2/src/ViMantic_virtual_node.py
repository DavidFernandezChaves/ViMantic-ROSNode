#! /usr/bin/env python
import sys

import cv2
import detectron2_ros.msg
import message_filters
import numpy as np
import rospy
import tf
import tf2_geometry_msgs
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point32, PoseStamped, Point, Vector3, Quaternion
from numpy import savetxt
from vimantic.msg import SemanticObject, SemanticObjects
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import PointCloud


class SemanticMappingNode(object):
    def __init__(self):

        # Topics
        self._threshold = rospy.get_param('~threshold', 0.5)
        self._debug = rospy.get_param('~debug', False)
        self._point_cloud_enabled = rospy.get_param('~point_cloud', False)
        self._publish_rate = rospy.get_param('~publish_rate', 100)

        # General Variables
        self._bridge = CvBridge()
        self._tfBuffer = tf2_ros.Buffer()
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
        self._pub_result = rospy.Publisher(rospy.get_param('~topic_result', 'vimantic/SemanticObjects'),
                                           SemanticObjects,
                                           queue_size=10)

        self._pub_repub = rospy.Publisher('vimantic/Camera-RGB', Image, queue_size=10)
        self._pub_repub2 = rospy.Publisher('vimantic/Camera-Depth', Image,queue_size=10)
        self._pub_pose = rospy.Publisher('vimantic/point', PoseStamped, queue_size=10)

        # Subscribers
        rospy.Subscriber(rospy.get_param('~topic_cnn'), detectron2_ros.msg.Result, self.callback_new_detection)
        self.subscriber = rospy.Subscriber("vimantic/virtualCamera", CompressedImage, self.callbackVirtualImage, queue_size=1)

        tf2_ros.TransformListener(self._tfBuffer)

    def run(self):

        rate = rospy.Rate(self._publish_rate)

        while not rospy.is_shutdown():
            # Republish last img
            if self._last_img_rgb is not None:
                self._pub_repub.publish(self._bridge.cv2_to_imgmsg(self._last_img_rgb, 'rgb8'))

            if self._cnn_msg is not None:
                # The detected objects are processed
                # if len(self._cnn_msg.class_names) > 0:
                #
                #     if self._enableTimeCapture:
                #         self._time_objectInfoPacket = rospy.get_rostime()
                #
                #     img_depth = self._last_msg[0]
                #     data_header = self._last_msg[1]
                #     data_transform = self._last_msg[2]
                #
                #     # Transform the value of each px to m by acquiring a cloud of points
                #     img_depth = img_depth / 6553.5
                #     img_depth = self.rotate_image(img_depth, self._img_angle)
                #
                #     rows, cols = img_depth.shape
                #     c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
                #
                #     z = img_depth
                #     x = ((self._cx - c) * z / self._fx)
                #     y = ((self._cy - r) * z / self._fy)
                #
                #     # Cut out every object from the point cloud and build the result.
                #     result = SemanticObjects()
                #
                #     result.header = data_header
                #     result.header.frame_id = "/map"
                #
                #     for i in range(len(self._cnn_msg.class_names)):
                #
                #         if self._cnn_msg.scores[i] > self._threshold:
                #             semantic_object = SemanticObject()
                #             point_cloud = PointCloud()
                #             point_cloud.header = data_header
                #
                #             semantic_object.score = self._cnn_msg.scores[i]
                #             semantic_object.type = self._cnn_msg.class_names[i]
                #
                #             try:
                #                 mask = (self._bridge.imgmsg_to_cv2(self._cnn_msg.masks[i]) == 255)
                #             except CvBridgeError as e:
                #                 print(e)
                #
                #             if x.shape != mask.shape:
                #                 print(x.shape)
                #                 print(mask.shape)
                #
                #             else:
                #
                #                 x_ = x[mask]
                #                 y_ = y[mask]
                #                 z_ = z[mask]
                #
                #                 # Bandpass filter with Z data
                #                 top_margin = (z_.max() - z_.min()) * 0.9 + z_.min()
                #                 bottom_margin = (z_.max() - z_.min()) * 0.1 + z_.min()
                #
                #                 mask2 = np.logical_and(z_ > bottom_margin, z_ < top_margin)
                #
                #                 x_ = x_[mask2]
                #                 y_ = y_[mask2]
                #                 z_ = z_[mask2]
                #
                #                 if len(x_) == 0:
                #                     continue
                #
                #                 # point_cloud.channels = [ChannelFloat32("red", img_rgb[mask, 0]),
                #                 #                        ChannelFloat32("green", img_rgb[mask, 1]),
                #                 #                        ChannelFloat32("blue", img_rgb[mask, 2])]
                #
                #                 scale_x = x_.max() - x_.min()
                #                 scale_y = y_.max() - y_.min()
                #                 scale_z = np.std(z_)
                #
                #                 semantic_object.scale = Vector3(scale_x, scale_y, scale_z)
                #
                #                 # Calculate the center px
                #                 x_center = int(self._cnn_msg.boxes[i].x_offset + self._cnn_msg.boxes[i].width / 2)
                #                 y_center = int(self._cnn_msg.boxes[i].y_offset + self._cnn_msg.boxes[i].height / 2)
                #                 # And the depth of the center
                #                 z_center = -(float(scale_z / 2) + np.average(z_))
                #
                #                 # Transformed the center of the object to the map reference system
                #                 p1 = PoseStamped()
                #                 p1.header = data_header
                #
                #                 p1.pose.position = Point(-x[y_center, x_center], y[y_center, x_center], z_center)
                #                 p1.pose.orientation.w = 1.0  # Neutral orientation
                #                 res = tf2_geometry_msgs.do_transform_pose(p1, data_transform)
                #                 semantic_object.pose = res.pose
                #
                #                 self._pub_pose.publish(res)
                #
                #                 if self._point_cloud_enabled:
                #                     for j in range(len(z_)):
                #                         point_cloud.points.append(
                #                             Point32(-round(x_[j] - x_center, 4), round(y_[j] - y_center, 4),
                #                                     -round(z_[j] - z_center, 4)))
                #
                #                 semantic_object.pointCloud = point_cloud
                #                 result.semanticObjects.append(semantic_object)
                #
                #                 # Debug----------------------------------------------------------------------------------------
                #                 if self._debug:
                #                     print (self._cnn_msg.class_names[i] + ": " + str(self._cnn_msg.scores[i]))
                #                 # ---------------------------------------------------------------------------------------------
                #
                #     self._pub_result.publish(result)
                #
                #     if self._enableTimeCapture:
                #         self._list_time_objectInfoPacket.append(
                #             (rospy.get_rostime() - self._time_objectInfoPacket).nsecs / 1000000)

                self._cnn_msg = None
                self._waiting_cnn = False

            rate.sleep()

        if self._enableTimeCapture:
            savetxt('~/ViMantic/time_cnn.csv', self._list_time_cnn, delimiter=',')
            savetxt('~/ViMantic/time_objectInfoPacket.csv', self._list_time_objectInfoPacket, delimiter=',')

    # def _image_callback(self, depth_msg, rgb_msg):
    #
    #     if not self._waiting_cnn:
    #         # and self._msg_lock.acquire(False):
    #         try:
    #             img_rgb = self._bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
    #             img_depth = self._bridge.imgmsg_to_cv2(depth_msg, "16UC1")
    #         except CvBridgeError as e:
    #             print(e)
    #
    #         self._last_img_rgb = self.rotate_image(img_rgb, self._img_angle)
    #
    #         transform = self._tfBuffer.lookup_transform("map",
    #                                                     rgb_msg.header.frame_id,  # source frame
    #                                                     rospy.Time(0),  # get the tf at first available time
    #                                                     rospy.Duration(5))
    #
    #         # Robot@Home fixe
    #         #rotation = (transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z,
    #         #            transform.transform.rotation.w)
    #         #rot = tf.transformations.euler_from_quaternion(rotation)
    #         #newrot = tf.transformations.quaternion_from_euler(rot[0], rot[1], rot[2] - 90)
    #         #transform.transform.rotation = Quaternion(newrot[0], newrot[1], newrot[2], newrot[3])
    #
    #         self._last_msg = [img_depth, depth_msg.header, transform]
    #         self._waiting_cnn = True
    #         if self._enableTimeCapture:
    #             self._time_cnn = rospy.get_rostime()

    def callbackVirtualImage(self, ros_data):
        # self._bridge.comp
        np_arr = np.fromstring(ros_data.data, np.uint8)
        im = cv2.imdecode(np_arr, -1)
        imrgb = cv2.cvtColor(im[:,:,:3],cv2.COLOR_RGB2BGR)
        self._pub_repub.publish(self._bridge.cv2_to_imgmsg(imrgb, 'bgr8'))
        self._pub_repub2.publish(self._bridge.cv2_to_imgmsg(im[:,:,3], 'passthrough'))

    def callback_new_detection(self, result_cnn):
        if self._waiting_cnn and self._cnn_msg is None:
            self._cnn_msg = result_cnn
            # CNN Time
            if self._enableTimeCapture:
                self._list_time_cnn.append((rospy.get_rostime() - self._time_cnn).nsecs / 1000000)



def main(argv):
    rospy.init_node('semantic_mapping')
    node = SemanticMappingNode()
    node.run()


if __name__ == '__main__':
    main(sys.argv)
