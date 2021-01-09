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
        # Camera calibration
        self.cx = 320
        self.cy = 240
        self.fx = 457.1429
        self.fy = 470.5882

        # Topics
        self.threshold = rospy.get_param('~threshold', 0.5)
        self.debug = rospy.get_param('~debug', False)
        self.point_cloud_enabled = rospy.get_param('~point_cloud', False)
        self.publish_rate = rospy.get_param('~publish_rate', 100)

        # General Variables
        self.bridge = CvBridge()
        self.tfBuffer = tf2_ros.Buffer()
        self.last_package = None
        self.cnn_msg = None
        self.waiting_cnn = False

        # Time variables
        self.enableTimeCapture = rospy.get_param('~record_times', False)
        self.time_cnn = 0
        self.time_objectInfoPacket = 0
        self.list_time_cnn = []
        self.list_time_objectInfoPacket = []

        # Publisher
        self.pub_result = rospy.Publisher(rospy.get_param('~topic_result', 'vimantic/SemanticObjects'),
                                           SemanticObjects,
                                           queue_size=0)

        self.pub_repub = rospy.Publisher('vimantic/imgRGBToCNN', Image, queue_size=0)
        self.pub_pose = rospy.Publisher('vimantic/point', PoseStamped, queue_size=0)

        # Subscribers
        rospy.Subscriber(rospy.get_param('~topic_cnn'), detectron2_ros.msg.Result, self.callback_new_detection, queue_size=0)
        rospy.Subscriber("vimantic/virtualCameraRGBD", CompressedImage, self.callbackVirtualImage, queue_size=0)

        tf2_ros.TransformListener(self.tfBuffer)

    def run(self):

        rate = rospy.Rate(self.publish_rate)

        while not rospy.is_shutdown():
            # Republish last img
            if self.last_package is not None:
                self.pub_repub.publish(self.bridge.cv2_to_imgmsg(self.last_package[2], 'rgb8'))

            if self.cnn_msg is not None:
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
                    result = SemanticObjects()

                    result.header = data_header

                    for i in range(len(self.cnn_msg.class_names)):

                        if self.cnn_msg.scores[i] > self.threshold:
                            semantic_object = SemanticObject()
                            point_cloud = PointCloud()
                            point_cloud.header = data_header

                            semantic_object.score = self.cnn_msg.scores[i]
                            semantic_object.type = self.cnn_msg.class_names[i]

                            try:
                                mask = (self.bridge.imgmsg_to_cv2(self.cnn_msg.masks[i]) == 255)
                            except CvBridgeError as e:
                                print(e)

                            if x.shape != mask.shape:
                                print(x.shape)
                                print(mask.shape)
                            else:
                                x_ = x[mask]
                                y_ = y[mask]
                                z_ = z[mask]

                                # Bandpass filter with Z data
                                top_margin = (z_.max() - z_.min()) * 0.9 + z_.min()
                                # bottom_margin = (z_.max() - z_.min()) * 0.1 + z_.min()

                                # mask2 = np.logical_and(z_ > bottom_margin, z_ < top_margin)
                                mask2 = z_ < top_margin

                                x_ = x_[mask2]
                                y_ = y_[mask2]
                                z_ = z_[mask2]

                                if len(x_) == 0:
                                    continue

                                # point_cloud.channels = [ChannelFloat32("red", img_rgb[mask, 0]),
                                #                        ChannelFloat32("green", img_rgb[mask, 1]),
                                #                        ChannelFloat32("blue", img_rgb[mask, 2])]

                                scale_x = x_.max() - x_.min()
                                scale_y = y_.max() - y_.min()
                                scale_z = z_.max() - z_.min()

                                semantic_object.scale = Vector3(scale_x, scale_z, scale_y)

                                # Calculate the center
                                x_center = int(self.cnn_msg.boxes[i].x_offset + self.cnn_msg.boxes[i].width / 2)
                                y_center = int(self.cnn_msg.boxes[i].y_offset + self.cnn_msg.boxes[i].height / 2)
                                z_center = scale_z/2 + z_.min()

                                # print(x_center)
                                # print (y_center)
                                # print (z_center)

                                # Transformed the center of the object to the map reference system
                                p1 = PoseStamped()
                                p1.header = data_header
                                #
                                p1.pose.position = Point(z_center, x[y_center, x_center],y[y_center, x_center])
                                # print(p1.pose.position)
                                p1.pose.orientation.w = 1.0  # Neutral orientation
                                res = tf2_geometry_msgs.do_transform_pose(p1, data_transform)
                                # print(res.pose)
                                semantic_object.pose = res.pose
                                #semantic_object.pose = p1.pose

                                self.pub_pose.publish(res)

                                if self.point_cloud_enabled:
                                    for j in range(len(z_)):
                                        point_cloud.points.append(
                                            Point32(-round(x_[j] - x_center, 4), round(y_[j] - y_center, 4),
                                                    -round(z_[j] - z_center, 4)))

                                semantic_object.pointCloud = point_cloud
                                result.semanticObjects.append(semantic_object)

                                # Debug----------------------------------------------------------------------------------------
                                if self.debug:
                                    print (self.cnn_msg.class_names[i] + ": " + str(self.cnn_msg.scores[i]))
                                # ---------------------------------------------------------------------------------------------

                    self.pub_result.publish(result)

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
            np_arr = np.fromstring(img_msg.data, np.uint8)
            im = cv2.imdecode(np_arr, -1)
            img_rgb = cv2.cvtColor(im[:,:,:3],cv2.COLOR_RGB2BGR)

            img_depth = np.divide(im[:,:,3],255.0)

            transform = self.tfBuffer.lookup_transform("map",
                                                        img_msg.header.frame_id,  # source frame
                                                        rospy.Time(0),  # get the tf at first available time
                                                        rospy.Duration(5))

            self.last_package = [img_msg.header, transform, img_rgb, img_depth]
            self.waiting_cnn = True
            if self.enableTimeCapture:
                self.time_cnn = rospy.get_rostime()

            # self._pub_repub.publish(self._bridge.cv2_to_imgmsg(imrgb, 'bgr8'))
            # self._pub_repub2.publish(self._bridge.cv2_to_imgmsg(im[:,:,3], 'passthrough'))

    def callback_new_detection(self, result_cnn):
        if self.waiting_cnn and self.cnn_msg is None:
            self.cnn_msg = result_cnn
            # CNN Time
            if self.enableTimeCapture:
                self.list_time_cnn.append((rospy.get_rostime() - self.time_cnn).nsecs / 1000000)



def main(argv):
    rospy.init_node('semantic_mapping')
    node = SemanticMappingNode()
    node.run()


if __name__ == '__main__':
    main(sys.argv)
