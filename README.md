# ViMantic Client (ROS)
ViMantic is a distributed architecture for semantic mapping of environments using mobile robots. For this, we have used Unity to create virtual environments that represent the information obtained from the real environment. This architecture is composed of one or several clients (robots/agents) and a server.

## Features
- Use an ontology as a formal and clear model to accommodate semantic information, including also mechanisms for its manipulation, i.e. insertion, modification or query.
- The model is automatically populated, i.e. it has a method to transform sensory data into high-level information, e.g. by recognizing objects.
- ViMantic uses 3D virtual maps to represent the semantic knowledge acquired.

## Requirements
- [ViMantic - Server](https://github.com/DavidFernandezChaves/ViMantic-Server)
- [RosBridge](http://wiki.ros.org/rosbridge_suite)
- [Detectron2](https://github.com/DavidFernandezChaves/Detectron2_ros) (Replaceable) 

This software uses [Detectron2](https://github.com/DavidFernandezChaves/Detectron2_ros) as object recognizer. However, this architecture can be easily modified to use other object recognition systems.

## Process
This node uses the Detectron2 to recognize objects in RGB images. It then extracts the point clouds of each detected object from a depth image. The point clouds are processed to try to fit the object and eliminate spurious points. Finally the information of the detected objects is sent to the architecture server.

## Parameters
```bash
        #Name of the topic where the results are published
        <param name="topic_result" value="ViMantic/Detections"/>
        
        #Topic name where the RGB image is obtained
	<param name="topic_intensity" value="RGBD_4_intensity"/>
	    
        #Topic name where the depth image is obtained
        <param name="topic_depth" value="RGBD_4_depth"/>
        
        #Topic name of CNN input
        <param name="topic_republic" value="ViMantic/ToCNN"/>	
	
	#Minimum object size 
        <param name="min_size" value="0.05"/>
        
        #Topic name of CNN results
        <param name="topic_cnn" value="detectron2_ros/result"/>       
        
        #Enables debug mode
	<param name="debug" value="true"/>
```
