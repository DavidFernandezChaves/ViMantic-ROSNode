#Define a 3D detection in ViMantic.

ObjectHypothesis[] scores

# ViMantic semantic knowledge:
# 0 -> top left // 1 -> top right // 2 -> down right // 3 -> down left
geometry_msgs/Point[] corners

# Height value:
float64 height

# Fixed corners:
uint8 fixed_corners