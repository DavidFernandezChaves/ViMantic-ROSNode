#! /usr/bin/env python
import sys
import time
import cv2
import numpy as np
import rospy
import tf2_geometry_msgs
import message_filters
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, PoseWithCovariance, Point, Vector3
from sensor_msgs.msg import Image, CompressedImage
from vimantic.msg import SemanticObject, SemanticObjectArray
from vision_msgs.msg import Detection2DArray, ObjectHypothesis


class ViManticNode(object):
    def __init__(self):
        rospy.logwarn("Initializing")
        # ROS Parameters
        self.image_rgb_topic = self.load_param('~topic_virtualCameraRGB', "ViMantic/virtualCameraRGB")
        self.image_depth_topic = self.load_param('~topic_virtualCameraDepth', "ViMantic/virtualCameraDepth")
        self.semantic_topic = self.load_param('~topic_result', 'ViMantic/SemanticObjects')
        self.cnn_topic = self.load_param('~topic_cnn')
        self.image_toCNN = self.load_param('~topic_republic', 'ViMantic/ToCNN')
        self.threshold = self.load_param('~threshold', 0.5)
        self.debug = self.load_param('~debug', False)

        # Camera calibration
        self._cx = 320
        self._cy = 240
        self._fx = 457.1429
        self._fy = 470.5882

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
        rospy.Subscriber(self.cnn_topic, Detection2DArray, self.callback_cnn, queue_size=1)

        sub_rgb_image = message_filters.Subscriber(self.image_rgb_topic, CompressedImage)
        sub_depth_image = message_filters.Subscriber(self.image_depth_topic, CompressedImage)

        message_filter = message_filters.ApproximateTimeSynchronizer([sub_depth_image, sub_rgb_image], 10, 0.3)
        message_filter.registerCallback(self.callbackVirtualImage)

        tf2_ros.TransformListener(self._tfBuffer)
        self.start_time = 0
        rospy.logwarn("Initialized")

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():

            # Republish last img
            # if self._waiting_cnn:
            #     self._pub_repub.publish(self._bridge.cv2_to_imgmsg(self._last_msg[1], 'rgb8'))

            if self._waiting_cnn and self._tries > 200:
                self._waiting_cnn = False
                rospy.logwarn("[ViMantic] CNN does not respond.")
            else:
                self._tries += 1

            rate.sleep()

    def px2cm(self, px, py, depth):
        x = ((self._cx - px) * depth[int(py) - 1, int(px) - 1] / self._fx)
        y = ((self._cy - py) * depth[int(py) - 1, int(px) - 1] / self._fy)

        return [x, y]

    def callbackVirtualImage(self, depth_msg, rgb_msg):

        if not self._waiting_cnn:
            transform = self._tfBuffer.lookup_transform("map",
                                                        rgb_msg.header.frame_id,  # source frame
                                                        rospy.Time(0),  # get the tf at first available time
                                                        rospy.Duration(5))

            img_rgb = self.decode_image_rgb_from_unity(rgb_msg.data)
            img_depth = self.decode_image_depth_from_unity(depth_msg.data)

            self._last_msg = [rgb_msg.header, img_rgb, img_depth, transform]
            self._pub_repub.publish(self._bridge.cv2_to_imgmsg(self._last_msg[1], 'rgb8'))
            self._tries = 0
            self._waiting_cnn = True
            if self.start_time == 0:
                self.start_time = time.time()
        else:
            rospy.logwarn("Image loss")

    def callback_cnn(self, result_cnn):
        if self._waiting_cnn:

            self._image_counter = self._image_counter + 1
            if (self._image_counter % 11) == 10:
                rospy.loginfo("Images detected per second=%.2f",
                              float(self._image_counter) / (time.time() - self.start_time))

            if len(result_cnn.detections) > 0:
                data_header = self._last_msg[0]
                img_depth = self._last_msg[2]
                data_transform = self._last_msg[3]

                # Transform the value of each px to m by acquiring a cloud of points
                img_depth = img_depth * 15

                result = SemanticObjectArray()

                result.header = data_header
                result.header.frame_id = "/map"
                objstring = 'Detected:'
                detections = result_cnn.detections
                detections.sort(key=lambda x: x.bbox)

                index = 0

                while index < len(detections):

                    if detections[index].results[0].score > self.threshold:
                        semanticObject = SemanticObject()

                        box = detections[index].bbox

                        try:
                            z_clipping_center = img_depth[int(box.center.y - 5):int(box.center.y + 5),
                                                int(box.center.x - 5):int(box.center.x + 5)]

                            z_clipping_box = img_depth[
                                             int(box.center.y + 5 - box.size_y / 2):int(
                                                 box.center.y - 5 + box.size_y / 2),
                                             int(box.center.x + 5 - box.size_x / 2):int(
                                                 box.center.x - 5 + box.size_x / 2)]
                        except:
                            rospy.logwarn("Error extracting the z_clipping_center")
                            break

                        # print(box)
                        # print(img_depth.shape)
                        # print(z_clipping_box.shape)
                        # print(str(box.center.y + 5 - box.size_y / 2) + ":" + str(
                        #     box.center.y - 5 + box.size_y / 2) + "/" + str(
                        #     box.center.x + 5 - box.size_x / 2) + ":" + str(box.center.x - 5 + box.size_x / 2))

                        shape = z_clipping_center.shape
                        if shape[0] != 10 or shape[1] != 10:
                            rospy.logwarn("Error in the form of z_clipping_center ")
                            break

                        [x_min, y_min] = self.px2cm(int(box.center.x - box.size_x / 2),
                                                    int(box.center.y - box.size_y / 2),
                                                    img_depth)
                        [x_max, y_max] = self.px2cm(int(box.center.x + box.size_x / 2),
                                                    int(box.center.y + box.size_y / 2),
                                                    img_depth)

                        # Calculate size
                        scale_x = abs(x_max - x_min)
                        scale_y = abs(y_max - y_min)
                        # scale_z = abs(z_clipping_box.max() - z_clipping_box.min())

                        # Fixing z scale
                        if scale_x > scale_y:
                            scale_z = abs(scale_y)
                        else:
                            scale_z = abs(scale_x)

                        # Bandpass filter with Z data
                        # top_margin = (img_depth.max() - img_depth.min()) * 0.9 + z_clipping_center.min()
                        # bottom_margin = (z_clipping_center.max() - z_clipping_center.min()) * 0.1 + z_clipping_center.min()
                        #
                        # mask2 = np.logical_and(z_clipping_center > bottom_margin, z_clipping_center < top_margin)
                        #
                        # z_clipping_center = z_clipping_center[mask2]

                        semanticObject.size = Vector3(scale_x, scale_z, scale_y)

                        # Calculate the center
                        [x_center, y_center] = self.px2cm(box.center.x, box.center.y, img_depth)
                        z_center = np.average(z_clipping_center)

                        # Transformed the center of the object to the map reference system
                        p1 = PoseStamped()
                        p1.header = data_header

                        # p1.pose.position = Point(-x_center, y_center, -z_center)
                        p1.pose.position = Point(z_center, x_center, y_center)
                        p1.pose.orientation.w = 1.0  # Neutral orientation
                        ans = tf2_geometry_msgs.do_transform_pose(p1, data_transform)

                        semanticObject.pose = PoseWithCovariance()
                        semanticObject.pose.pose = ans.pose
                        det = ObjectHypothesis()
                        det.id = detections[index].results[0].id
                        det.score = detections[index].results[0].score

                        semanticObject.scores.append(det)

                        objstring = objstring + ' [' + det.id + ', p=%.2f' % det.score

                        while (index + 1) < len(detections) and detections[index].bbox == detections[index + 1].bbox:
                            index += 1
                            objstring = objstring + ' / '
                            det = ObjectHypothesis()
                            det.id = detections[index].results[0].id
                            det.score = detections[index].results[0].score
                            semanticObject.scores.append(det)
                            objstring = objstring + det.id + ', p=%.2f' % det.score

                        result.semanticObjects.append(semanticObject)
                        objstring = objstring + '] '

                    index += 1

                if len(result.semanticObjects) > 0:
                    self._pub_pose.publish(ans)
                    self._pub_result.publish(result)
                    rospy.loginfo(objstring)

            self._waiting_cnn = False

    @staticmethod
    def load_param(param, default=None):
        new_param = rospy.get_param(param, default)
        rospy.loginfo("[ViMantic] %s: %s", param, new_param)
        return new_param

    @staticmethod
    def decode_image_rgb_from_unity(unity_img):
        np_arr = np.fromstring(unity_img, np.uint8)
        im = cv2.imdecode(np_arr, -1)
        img_rgb = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        return img_rgb

    @staticmethod
    def decode_image_depth_from_unity(unity_img):
        np_arr = np.fromstring(unity_img, np.uint8)
        im = cv2.imdecode(np_arr, -1)
        img_depth = np.divide(im, 255.0)
        img_depth = cv2.cvtColor(img_depth, cv2.COLOR_RGB2GRAY)

        return img_depth


if __name__ == '__main__':
    rospy.init_node("ViMantic", anonymous=False, log_level=rospy.INFO)
    node = ViManticNode()
    node.run()
    rospy.spin()
