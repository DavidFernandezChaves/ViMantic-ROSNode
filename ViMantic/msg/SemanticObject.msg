#Define a 3D object with semantic information in ViMantic.

string id

ObjectHypothesis[] scores

#ViMantic semantic knowledge:
geometry_msgs/Pose pose

#ID in the ontology of the room in which the object is located.
string roomId

#Number of times this object has been recognised by the system
uint32 detections

#Type of the room in which the object is located.
string roomType

# 3D object size
geometry_msgs/Vector3 size
