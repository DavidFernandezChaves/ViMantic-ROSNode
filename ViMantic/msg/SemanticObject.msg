#Define a 3D object with semantic information in ViMantic.

#ID of the object in the ontology (empty if not known).
vision_msgs/ObjectHypothesisWithPose object

#ViMantic semantic knowledge:
#Estimated object type according to the object recognizer.
string objectType

#ID in the ontology of the room in which the object is located.
string roomId

#Type of the room in which the object is located.
string roomtype

# 3D object size
geometry_msgs/Vector3 size
