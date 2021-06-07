#! /usr/bin/env python
import sys
import time

import cv2
import detectron2_ros.msg
import numpy as np
import open3d as o3d
import rospy
import tf2_geometry_msgs
import tf2_ros
import matplotlib.cm as plt
from math import pi, sin, cos
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import quaternion_from_euler

from geometry_msgs.msg import PoseStamped, Point, Vector3, Pose, Quaternion
from sensor_msgs.msg import Image, CompressedImage
from vimantic.msg import Detection, DetectionArray, ObjectHypothesis

class ViManticNode(object):
    def __init__(self):

        # Control and flags
        self._publish_rate = 10
        self._image_counter = 0
        self._start_time = 0
        self._tries = 0
        self._max_tries = 10
        self._flag_processing = False
        self._flag_cnn = False

        rospy.logwarn("Initializing")
        # ROS Parameters
        self.image_topic = self.load_param('~topic_virtualCameraRGBD', "ViMantic/virtualCameraRGBD")
        self.semantic_topic = self.load_param('~topic_result', 'ViMantic/Detections')
        self.cnn_topic = self.load_param('~topic_cnn', 'detectron2_ros/result')
        self.image_toCNN = self.load_param('~topic_republic', 'ViMantic/ToCNN')
        self.n_steps_fitting = self.load_param('~n_steps_fitting', 90)
        self._min_size = self.load_param('~min_size', 0.05)
        self.debug = self.load_param('~debug', False)

        # Orientation Fitting Variables
        theta = (89.00 / self.n_steps_fitting) * pi / 180.0
        self._R = self.y_rotation(theta)

        # Camera Calibration
        self._cx = 320
        self._cy = 240
        self._fx = 457.1429
        self._fy = 470.5882
        self._depth_range = 15
        self._max_distance_obj = 10 # Maximum distance to observe an object (in meters)

        # General Variables
        self._last_msg = None
        self._last_cnn_result = None
        self._image_r = None
        self._image_c = None

        # Publishers
        self._pub_result = rospy.Publisher(self.semantic_topic, DetectionArray, queue_size=10)
        self._pub_processed_image = rospy.Publisher(self.image_toCNN, Image, queue_size=1)

        # Subscribers
        rospy.Subscriber(self.cnn_topic, detectron2_ros.msg.Result, self.callback_new_detection)
        rospy.Subscriber(self.image_topic, CompressedImage, self.callback_virtual_image, queue_size=10)

        # Handlers
        self._bridge = CvBridge()
        self._tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self._tfBuffer)

        rospy.logwarn("Initialized")

    def display_inlier_outlier(self, cloud, ind):
        inlier_cloud = cloud.select_down_sample(ind)
        outlier_cloud = cloud.select_down_sample(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    def run(self):

        rate = rospy.Rate(self._publish_rate)

        while not rospy.is_shutdown():

            # Extracting and publishing detected objects
            if self._flag_processing and self._flag_cnn:

                # Obtain 3D coordinates in meters of each pixel
                z = self._last_msg[2] * self._depth_range
                x = ((self._cx - self._image_c) * z / self._fx)
                y = ((self._cy - self._image_r) * z / self._fy)

                # Initializing result message
                detections = DetectionArray()
                detections.header = self._last_msg[0]
                detections.header.frame_id = "/map"
                detections.origin = self._last_msg[4]

                # Initializing console message
                obj_string = 'Detected:'

                for idx in range(len(self._last_cnn_result.class_names)):

                    # Remove person objects
                    if self._last_cnn_result.class_names[idx] == "person":
                        continue

                    detection = Detection()
                    det = ObjectHypothesis()
                    det.id = self._last_cnn_result.class_names[idx]
                    det.score = self._last_cnn_result.scores[idx]
                    detection.scores.append(det)

                    try:
                        mask = (self._bridge.imgmsg_to_cv2(self._last_cnn_result.masks[idx]) == 255)

                    except CvBridgeError as e:
                        print(e)
                        continue

                    #kernel = np.ones((10,10),np.uint8)
                    #mask = cv2.erode(np.float32(mask),kernel).astype(bool)

                    if np.sum(mask) == 0:
                        continue

                    # Get detected object point cloud
                    x_ = x[mask]
                    y_ = y[mask]
                    z_ = z[mask]

                    # Statistics from Z data
                    mean = np.mean(z_)
                    std = np.std(z_)

                    # Bandpass filter with Z data
                    top_margin = mean + 1.5 * std
                    bottom_margin = mean - 1.5 * std
                    filtered_mask = np.logical_and(z_ > bottom_margin, z_ < top_margin)

                    if np.sum(filtered_mask) == 0:
                        continue

                    # Obtain filtered point cloud
                    point_cloud = np.array([x_[filtered_mask].reshape(-1),
                                            y_[filtered_mask].reshape(-1),
                                            z_[filtered_mask].reshape(-1)]).T

                    if np.mean(z_[filtered_mask]) > self._max_distance_obj:
                        continue

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(point_cloud)
                    _, ind = pcd.remove_radius_outlier(nb_points=30, radius=0.05)
                    # self.display_inlier_outlier(pcd, ind)
                    pcd = pcd.select_down_sample(ind)

                    if not pcd.has_points():
                        continue

                   # labels = np.array(pcd.cluster_dbscan(eps=0.006, min_points=10, print_progress=True))
                   #
                   #  max_label = labels.max()
                   #  colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
                   #  colors[labels < 0] = 0
                   #  pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
                   #  o3d.visualization.draw_geometries([pcd])

                    best_angle = 0
                    best_obb = pcd.get_axis_aligned_bounding_box()
                    volume = best_obb.volume()

                    aabb = pcd.get_axis_aligned_bounding_box()
                    aabb.color = (1, 0, 0)
                    # o3d.visualization.draw_geometries([pcd, aabb])

                    for i in range(self.n_steps_fitting):
                        pcd.rotate(self._R)

                        aabb = pcd.get_axis_aligned_bounding_box()
                        aabb.color = (1, 0, 0)

                        if aabb.volume() < volume:
                            best_obb = aabb
                            volume = aabb.volume()
                            best_angle = (i + 1) * 90.0 / self.n_steps_fitting

                    if detection.height[0] < self._min_size or detection.height[1] < self._min_size or detection.height[2] < self._min_size:
                        continue

                    detection.height = np.asarray(best_obb.get_extent())

                    best_obb = o3d.geometry.OrientedBoundingBox(best_obb.get_center(),
                                                                self.y_rotation(best_angle * pi / 180.0),
                                                                detection.height)

                    detection.corners = []

                    for pt in np.asarray(best_obb.get_box_points()):
                        if pt[1] > best_obb.get_center()[1]:
                            detection.corners.append(tf2_geometry_msgs.do_transform_point(Point(*pt),
                                                                                          self._last_msg[3]).point)

                    detections.detections.append(detection)
                    obj_string = obj_string + ' ' + det.id + ', p=%.2f.' % det.score

                self._pub_result.publish(detections)
                rospy.loginfo(obj_string)

                self._flag_cnn = False
                self._flag_processing = False

            # CNN does not respond.
            elif self._flag_processing and not self._flag_cnn:
                # Skipping frame.
                if self._tries > self._max_tries:
                    self._flag_processing = False
                    rospy.logwarn("[ViMantic] CNN does not respond, skipping frame.")

                # Awaiting CNN to process the last image
                else:
                    self._pub_processed_image.publish(self._bridge.cv2_to_imgmsg(self._last_msg[1], 'rgb8'))
                    self._tries += 1

            rate.sleep()

    def callback_virtual_image(self, img_msg):

        if not self._flag_processing:
            transform = self._tfBuffer.lookup_transform("map",
                                                        img_msg.header.frame_id,  # source frame
                                                        rospy.Time(0))  # get the tf at first available time

            img_rgb, img_depth = self.decode_image_from_unity(img_msg.data)

            try:
                trans = self._tfBuffer.lookup_transform('map', img_msg.header.frame_id, rospy.Time())
                origin = Pose()
                origin.position.x = trans.transform.translation.x
                origin.position.y = trans.transform.translation.y
                origin.position.z = trans.transform.translation.z
                origin.orientation = trans.transform.rotation

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn("Failing to retrieve robot pose")

            self._last_msg = [img_msg.header, img_rgb, img_depth, transform, origin]
            self._pub_processed_image.publish(self._bridge.cv2_to_imgmsg(self._last_msg[1], 'rgb8'))
            self._tries = 0

            if self._start_time == 0:
                self._start_time = time.time()

            if self._image_c is None and self._image_r is None:
                # Generate a meshgrid where each pixel contains its pixel coordinates
                rows, cols = img_depth.shape
                self._image_c, self._image_r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
            self._flag_processing = True

    def callback_new_detection(self, result_cnn):

        if self._flag_processing and not self._flag_cnn:
            self._image_counter = self._image_counter + 1
            if (self._image_counter % 11) == 10:
                rospy.loginfo("Images detected per second=%.2f",
                              float(self._image_counter) / (time.time() - self._start_time))

            if len(result_cnn.class_names) > 0:
                self._last_cnn_result = result_cnn
                self._flag_cnn = True

            else:
                self._flag_processing = False

    # Static Methods
    @staticmethod
    def load_param(param, default=None):
        new_param = rospy.get_param(param, default)
        rospy.loginfo("[ViMantic] %s: %s", param, new_param)
        return new_param

    @staticmethod
    def decode_image_from_unity(unity_img):
        np_arr = np.fromstring(unity_img, np.uint8)
        im = cv2.imdecode(np_arr, -1)
        img_rgb = cv2.cvtColor(im[:, :, :3], cv2.COLOR_RGB2BGR)
        img_depth = np.divide(im[:, :, 3], 255.0)

        return img_rgb, img_depth

    @staticmethod
    def y_rotation(theta):
        return np.asarray([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])


def main(argv):
    rospy.init_node('ViMantic')
    node = ViManticNode()
    node.run()


if __name__ == '__main__':
    main(sys.argv)
