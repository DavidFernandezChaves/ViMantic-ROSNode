^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package rr_mxnet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
0.2.1 (2018-07-19)
------------------
* Refactor segmentation code to move helper functions into custom_functions file and remodularize

0.2.0 (2018-10-4)
------------------
* SSD and Segmentation networks with GluonCV 0.3.0 and MxNet 1.3
* SSD utilize segmentation mask to restrict detections if available

0.1.0 (2018-09-26)
------------------
* first public release for Kinetic
* includes SSD (single-shot detector) node
* documentation on ros wiki rr_mxnet
* SSD based on mxnet/example/ssd and MxNet 1.1
* SSD helper functions to crop image into smaller chunks for tradeoff of pyramid sampling and single-shot detection to increase detection range
