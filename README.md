# ViMantic Client - ROS
ViMantic is a distributed architecture for the building of semantic maps using mobile robots. It has been designed as a client-server architecture that can run in different devices. It is characterized by:

- Use an ontology as a formal and clear model to accommodate semantic information, including also mechanisms for its manipulation, i.e. insertion, modification or query.
- The model is automatically populated, i.e. it has a method to transform sensory data into high-level information, e.g. by recognizing objects.
- ViMantic uses 3D virtual maps to show the semantic knowledge acquired. The user can use these maps to eliminate wrong knowledge in an intuitive way through a friendly interface.

<div align="center">
  <img src="https://github.com/DavidFernandezChaves/ViMantic-Server/blob/master/Imgs/Head.PNG?raw=true"/>
</div>

## Requirements
- [ViMantic - Server](https://github.com/DavidFernandezChaves/ViMantic-Server)
- [RosBridge](http://wiki.ros.org/rosbridge_suite)
- (Optional) [Robot@Home](http://mapir.isa.uma.es/mapirwebsite/index.php/mapir-downloads/203-robot-at-home-dataset.html) dataset.

This software is designed to be launched using [Detectron2](https://github.com/DavidFernandezChaves/Detectron2_ros) in ros. However, it can be easily modified to use a different CNN.

## Use
Use the launcher: ViMantic_Detectron2/launch/ViMantic.launch

Parameters:
```bash
        #Name of the topic where the results are published
        <param name="topic_result" value="semantic_mapping/SemanticObject"/>
        
        #Topic name where the RGB image is obtained
	    <param name="topic_intensity" value="RGBD_4_intensity"/>
	    
        #Topic name where the depth image is obtained
        <param name="topic_depth" value="RGBD_4_depth"/>
        
        #Topic name of CNN input
        <param name="topic_republic" value="semantic_mapping/RGB"/>
        
        #Topic name of CNN results
        <param name="topic_cnn" value="detectron2_ros/result"/>
        
        #Threshold of accuracy_estimation to publish a detected object        
        <param name="threshold" value="0.50"/>
        
        #Angle of the input image
        <param name="input_angle" value="90"/>
        
        #Enables the sending of the object's point cloud. (Disable in case of slow wireless networks)
        <param name="point_cloud" value="false"/>        
        
        #Enables debug mode
	    <param name="debug" value="true"/>
```

## Examples
<div align="center">
  <img src="https://github.com/DavidFernandezChaves/ViMantic-Server/blob/master/Imgs/maps.PNG?raw=true"/>
</div>
