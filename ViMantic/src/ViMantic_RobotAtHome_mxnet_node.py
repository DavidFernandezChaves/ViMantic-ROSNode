#! /usr/bin/env python
import time
import cv2
import message_filters
import numpy as np
import rospy
import tf2_geometry_msgs
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, PoseWithCovariance, Point, Vector3
from sensor_msgs.msg import Image
from vimantic.msg import SemanticObject, SemanticObjectArray
from vision_msgs.msg import Detection2DArray

class ViMantic:
    def __init__(self):
        rospy.logwarn("Initializing")

        # ROS Parameters
        rospy.loginfo("[ViMantic] Loading ROS Parameters")
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
        self._last_msg = None
        self._waiting_cnn = False
        self._tries = 0

        # Publisher
        self._pub_result = rospy.Publisher(self.semantic_topic, SemanticObjectArray, queue_size=10)
        self._pub_repub = rospy.Publisher(self.image_toCNN, Image, queue_size=1)

        if (self.debug):
            self._pub_pose = rospy.Publisher('ViMantic/detectedObject', PoseStamped, queue_size=1)

        # Subscribers
        rospy.Subscriber(self.cnn_topic, Detection2DArray, self.callback_cnn)
        sub_rgb_image = message_filters.Subscriber(self.image_rgb_topic, Image)
        sub_depth_image = message_filters.Subscriber(self.image_depth_topic, Image)

        message_filter = message_filters.ApproximateTimeSynchronizer([sub_depth_image, sub_rgb_image], 10, 0.3)
        message_filter.registerCallback(self.callback_img)

        tf2_ros.TransformListener(self._tfBuffer)
        self.start_time = time.time()
        rospy.logwarn("Initialized")

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():

            # Republish last img
            if self._waiting_cnn and self._tries > 50:
                self._pub_repub.publish(self._bridge.cv2_to_imgmsg(self._last_msg[1], 'rgb8'))
                self._waiting_cnn = False
                rospy.logwarn("[ViMantic] CNN does not respond, trying again.")
            else:
                self._tries += 1

            rate.sleep()

    def px2cm(self, px, py, depth):
        x = ((self._cx - px) * depth[int(py) - 1, int(px) - 1] / self._fx)
        y = ((self._cy - py) * depth[int(py) - 1, int(px) - 1] / self._fy)

        return [x, y]

    def callback_img(self, depth_msg, rgb_msg):

        if not self._waiting_cnn:
            try:
                img_rgb = self._bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
                img_depth = self._bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            except CvBridgeError as e:
                print(e)

            if self.input_angle != 0:
                img_rgb = self.rotate_image(img_rgb, self.input_angle)
                img_depth = self.rotate_image(img_depth, self.input_angle)

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

            self._last_msg = [rgb_msg.header, img_rgb, img_depth, transform]
            self._pub_repub.publish(self._bridge.cv2_to_imgmsg(self._last_msg[1], 'rgb8'))
            self._tries = 0
            self._waiting_cnn = True

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

                # Transform the value of each px to m
                img_depth = img_depth / 6553.5

                result = SemanticObjectArray()

                result.header = data_header
                result.header.frame_id = "/map"
                objstring = 'Detected:'
                for i in range(len(result_cnn.detections)):
                    semanticObject = SemanticObject()

                    semanticObject.object.score = result_cnn.detections[i].results[0].score
                    semanticObject.objectType = result_cnn.detections[i].results[0].id

                    box = result_cnn.detections[i].bbox

                    z_clipping = img_depth[int(box.center.x - box.size_x / 3):int(box.center.x + box.size_x / 3),
                                 int(box.center.y - box.size_y / 3):int(box.center.y + box.size_y / 3)]

                    if len(z_clipping) == 0:
                        return

                    # Bandpass filter with Z data
                    top_margin = (z_clipping.max() - z_clipping.min()) * 0.9 + z_clipping.min()
                    bottom_margin = (z_clipping.max() - z_clipping.min()) * 0.1 + z_clipping.min()

                    mask2 = np.logical_and(z_clipping > bottom_margin, z_clipping < top_margin)

                    z_clipping = z_clipping[mask2]

                    if len(z_clipping) == 0:
                        return

                    [x_min, y_min] = self.px2cm(int(box.center.x - box.size_x / 2), int(box.center.y - box.size_y / 2),
                                                img_depth)
                    [x_max, y_max] = self.px2cm(int(box.center.x + box.size_x / 2), int(box.center.y + box.size_y / 2),
                                                img_depth)

                    scale_x = abs(x_max - x_min)
                    scale_y = abs(y_max - y_min)
                    scale_z = abs(z_clipping.max() - z_clipping.min())

                    semanticObject.size = Vector3(scale_x, scale_y, scale_z)

                    # Calculate the center
                    [x_center, y_center] = self.px2cm(box.center.x, box.center.y, img_depth)
                    z_center = (float(scale_z / 2) + np.average(z_clipping))

                    # Transformed the center of the object to the map reference system
                    p1 = PoseStamped()
                    p1.header = data_header

                    p1.pose.position = Point(-x_center, y_center, -z_center)
                    p1.pose.orientation.w = 1.0  # Neutral orientation
                    ans = tf2_geometry_msgs.do_transform_pose(p1, data_transform)

                    semanticObject.object.pose = PoseWithCovariance()
                    semanticObject.object.pose.pose = ans.pose

                    self._pub_pose.publish(ans)

                    result.semanticObjects.append(semanticObject)
                    objstring = objstring + ' ' + semanticObject.objectType + ', p=%.2f.' % (
                        semanticObject.object.score)

                self._pub_result.publish(result)
                rospy.loginfo(objstring)

            self._waiting_cnn = False

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


if __name__ == '__main__':
    rospy.init_node("ViMantic", anonymous=False, log_level=rospy.INFO)
    node = ViMantic()
    rospy.spin()
