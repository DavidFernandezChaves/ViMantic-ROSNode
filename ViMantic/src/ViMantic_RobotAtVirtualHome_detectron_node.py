#! /usr/bin/env python
import sys
import time

import cv2
import detectron2_ros.msg
import numpy as np
import open3d as o3d
import rospy
import tf2_geometry_msgs
import message_filters
import tf2_ros
import matplotlib.cm as plt
from math import pi, sin, cos
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import quaternion_from_euler
from skimage.morphology import thin

from geometry_msgs.msg import PoseStamped, Point, Vector3, Pose, Quaternion, PointStamped
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
        self.image_rgb_topic = self.load_param('~topic_virtualCameraRGB', "ViMantic/virtualCameraRGB")
        self.image_depth_topic = self.load_param('~topic_virtualCameraDepth', "ViMantic/virtualCameraDepth")
        self.semantic_topic = self.load_param('~topic_result', 'ViMantic/Detections')
        self.cnn_topic = self.load_param('~topic_cnn', 'detectron2_ros/result')
        self.image_toCNN = self.load_param('~topic_republic', 'ViMantic/ToCNN')
        self.step_fitting = self.load_param('~step_fitting', 0.5)
        self._min_size = self.load_param('~min_size', 0.05)
        self.debug = self.load_param('~debug', False)

        # Camera Calibration
        self._cx = 320
        self._cy = 240
        self._fx = 548.5714
        self._fy = 568.4211
        #self._fx = 457.1429
        #self._fy = 470.5882
        self._depth_range = 15

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

        sub_rgb_image = message_filters.Subscriber(self.image_rgb_topic, CompressedImage)
        sub_depth_image = message_filters.Subscriber(self.image_depth_topic, CompressedImage)

        message_filter = message_filters.ApproximateTimeSynchronizer([sub_depth_image, sub_rgb_image], 10, 0.3)
        message_filter.registerCallback(self.callback_virtual_image)

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

                    #kernel = np.ones((3,3),np.uint8)
                    #mask = cv2.erode(np.float32(mask),kernel).astype(bool)
                    mask = thin(mask, 15)


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


                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(point_cloud)
                    #_, ind = pcd.remove_radius_outlier(nb_points=30, radius=0.05)
                    # self.display_inlier_outlier(pcd, ind)
                    #pcd = pcd.select_down_sample(ind)

                    if not pcd.has_points():
                        continue

                    # PointCloud Segmentation
                    # labels = np.array(pcd.cluster_dbscan(eps=0.006, min_points=10, print_progress=True))
                    #
                    #  max_label = labels.max()
                    #  colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
                    #  colors[labels < 0] = 0
                    #  pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
                    #  o3d.visualization.draw_geometries([pcd])

                    # Get best Angle
                    volume = pcd.get_axis_aligned_bounding_box().volume()
                    angle_step = self.step_fitting * pi / 180.0
                    pcd.rotate(self.y_rotation(angle_step))
                    best_angle = angle_step

                    best_obb = pcd.get_axis_aligned_bounding_box()
                    volume_turn = best_obb.volume()

                    if volume_turn > volume:
                        angle_step *= -1
                        volume = volume_turn

                    while volume_turn <= volume:
                        volume = volume_turn
                        pcd.rotate(self.y_rotation(angle_step))
                        volume_turn = pcd.get_axis_aligned_bounding_box().volume()
                        best_angle += angle_step

                    pcd.rotate(self.y_rotation(-angle_step))
                    best_obb = pcd.get_axis_aligned_bounding_box()
                    best_angle -= angle_step

                    scale = np.asarray(best_obb.get_extent())

                    if scale[0] < self._min_size or scale[1] < self._min_size or scale[2] < self._min_size:
                        continue

                    # o3d.visualization.draw_geometries([best_pcd, best_obb])
                    oriented_bb = o3d.geometry.OrientedBoundingBox(best_obb.get_center(),
                                                                   np.matmul(self.x_rotation(-10 * pi / 180.0),
                                                                             self.y_rotation(-best_angle)), scale)

                    detection.occluded_corners = 0
                    image = self._last_msg[1]
                    for i, pt in enumerate(np.asarray(oriented_bb.get_box_points())):
                        px = int(self._cx - (pt[0] * self._fx / pt[2]))
                        py = int(self._cy - (pt[1] * self._fy / pt[2]))

                        if px >= self._width - 30 or px <= 30 or py >= self._height - 31 or py <= 30:
                            # image = cv2.putText(image, str(i), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                            #                     cv2.LINE_AA)
                            detection.occluded_corners |= 1 << i
                            #image = cv2.circle(image, (px, py), 10, (0, 0, 255), -1)
                        elif pt[2] - 0.2 > z[py, px]:
                            detection.occluded_corners |= 1 << i
                            #image = cv2.circle(image, (px, py), 10, (0, 0, 255), -1)
                        #else:
                            #image = cv2.circle(image, (px, py), 10, (0, 255, 0), -1)

                    detection.occluded_corners |= ((detection.occluded_corners & (1 << 0)) << 3)
                    detection.occluded_corners |= ((detection.occluded_corners & (1 << 1)) << 5)
                    detection.occluded_corners |= ((detection.occluded_corners & (1 << 2)) << 3)
                    detection.occluded_corners |= ((detection.occluded_corners & (1 << 7)) >> 3)
                    detection.occluded_corners |= ((detection.occluded_corners & (1 << 0)) << 2)
                    detection.occluded_corners |= ((detection.occluded_corners & (1 << 1)) << 6)
                    """
                    for i, pt in enumerate(np.asarray(oriented_bb.get_box_points())):
                        px = int(self._cx - (pt[0] * self._fx / pt[2]))
                        py = int(self._cy - (pt[1] * self._fy / pt[2]))
                        if (detection.occluded_corners & (1 << i)) > 0:
                            image = cv2.circle(image, (px, py), 10, (0, 0, 255), -1)
                        else:
                            image = cv2.circle(image, (px, py), 10, (0, 255, 0), -1)
                    
                    image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
                    cv2.imshow("Paco",image)
                    cv2.waitKey(10)
                    """

                    # Transform local to global frame
                    # corners = []
                    detection.corners = []
                    for i, pt in enumerate(np.asarray(oriented_bb.get_box_points())):
                        # print("Punto original: " + str(pt))
                        global_point = tf2_geometry_msgs.do_transform_point(
                            PointStamped(self._last_msg[0], Point(pt[2], pt[0], pt[1])), self._last_msg[3]).point
                        detection.corners.append(global_point)

                    # Order corners
                    # corners.sort(key=lambda x: x[0].z, reverse=False)
                    # top_corners = corners[:4]
                    # bottom_corners = corners[4:]
                    #
                    # top_corners.sort(key=lambda x: x[0].x, reverse=False)
                    # bottom_corners.sort(key=lambda x: x[0].x, reverse=False)
                    #
                    # if top_corners[2][0].x - top_corners[1][0].x > 0.05 * scale[0]:
                    #     ordered_corners = top_corners[:2]
                    #     ordered_corners.sort(key=lambda x: x[0].y, reverse=False)
                    #     remain_corners = top_corners[2:4]
                    #     remain_corners.sort(key=lambda x: x[0].y, reverse=True)
                    #     ordered_corners += remain_corners
                    # else:
                    #     ordered_corners = top_corners[:3]
                    #     ordered_corners.sort(key=lambda x: x[0].y, reverse=False)
                    #     ordered_corners += [top_corners[3]]
                    #
                    # bottom_corners = [[Point(corner[0].x, corner[0].y, corner[0].z - scale[1]),corner[1]] for corner in
                    #                   ordered_corners]
                    #
                    # detection.corners = [pt[0] for pt in (ordered_corners + bottom_corners)]

                    # Get fixed corners
                    # detection.fixed_corners = 0
                    # for i, pt in enumerate(ordered_corners + bottom_corners):
                    #     detection.fixed_corners |= pt[1] << i

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

    def callback_virtual_image(self, depth_msg, rgb_msg):

        if not self._flag_processing:
            #transform = self._tfBuffer.lookup_transform("map",
            #                                            rgb_msg.header.frame_id,  # source frame
            #                                            rospy.Time(0))  # get the tf at first available time

            img_rgb = self.decode_image_rgb_from_unity(rgb_msg.data)
            img_depth = self.decode_image_depth_from_unity(depth_msg.data)

            try:
                trans = self._tfBuffer.lookup_transform('map', rgb_msg.header.frame_id, rospy.Time())
                origin = Pose()
                origin.position.x = trans.transform.translation.x
                origin.position.y = trans.transform.translation.y
                origin.position.z = trans.transform.translation.z
                origin.orientation = trans.transform.rotation

                self._last_msg = [rgb_msg.header, img_rgb, img_depth, trans, origin]
                self._pub_processed_image.publish(self._bridge.cv2_to_imgmsg(self._last_msg[1], 'rgb8'))
                self._tries = 0

                if self._start_time == 0:
                    self._start_time = time.time()

                if self._image_c is None and self._image_r is None:
                    # Generate a meshgrid where each pixel contains its pixel coordinates
                    self._height, self._width = img_depth.shape
                    self._image_c, self._image_r = np.meshgrid(np.arange(self._width), np.arange(self._height),
                                                               sparse=True)
                self._flag_processing = True

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn("Failing to retrieve robot pose")
                origin = Pose()



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

    @staticmethod
    def decode_image_from_unity(unity_img):
        np_arr = np.fromstring(unity_img, np.uint8)
        im = cv2.imdecode(np_arr, -1)
        img_rgb = cv2.cvtColor(im[:, :, :3], cv2.COLOR_RGB2BGR)
        img_depth = np.divide(im[:, :, 3], 255.0)

        return img_rgb, img_depth

    @staticmethod
    def x_rotation(theta):
        return np.asarray([[1, 0, 0], [0, cos(theta), -sin(theta)], [0, sin(theta), cos(theta)]])

    @staticmethod
    def y_rotation(theta):
        return np.asarray([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])

    @staticmethod
    def z_rotation(theta):
        return np.asarray([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])


def main(argv):
    rospy.init_node('ViMantic')
    node = ViManticNode()
    node.run()


if __name__ == '__main__':
    main(sys.argv)
