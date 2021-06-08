#Define a 3D detection in ViMantic.

ObjectHypothesis[] scores

# ViMantic semantic knowledge:
# 0 -> top left // 1 -> top right // 2 -> down right // 3 -> down left
geometry_msgs/Point[] corners

# Fixed corners:
uint8 fixed_corners