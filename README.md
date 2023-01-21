# PlaceRecognitionDB
PlaceRecognitionDB is a tool for creating optimally sized databases (containing the minimum number of frames covering the scene) for place recognition task from RGBD and LiDAR data.

Several methods of reducing the number of frames in databases have already been developed, as well as some metrics to evaluate the results of compression.

The goal of the project is to create a tool with an optimal compression strategy for any RGBD and LiDAR data.

## Datasets format
To use the tool, your data must be in a specific format.
* Undistorted color images in any format supported by OpenCV.
* Depth maps corresponding to color images in any format supported by Open3D.
* The trajectory containing one pose in each line in `timestamp tx ty tz qx qy qz qw` format.

Therefore, the structure of the dataset should look like this:
```
Example dataset
├── rgb_images
|   ├── 001.png
|   ├── 002.png
|   ├── ...
├── point_clouds
|   ├── 001.pcd
|   ├── 002.pcd
|   ├── ...
└── trajectory.txt
```
The number of images, point clouds and poses in the trajectory file must be the same.

## Example of usage
```
# Imports
from src.io import Exporter, read_dataset
from src.reduction_methods import SetCover

# Read dataset
database = read_dataset(path_to_dataset, pcd_folder_name, 
                        rgb_folder_name, traj_file_name)
                        
# Create reduced DB                        
reduced_db = SetCover(number_of_images_in_db).reduce(database)

# Export the created DB to the hard drive
Exporter().export(path_to_save, reduced_db)
```

## Repository structure
```
place-recognition-db
├── src — code of the tool
|   ├── core
|   ├── cos_place — provider to access the CosPlase neural network
|   ├── io — contains instruments for loading and exporting data
|   ├── providers — providers for point clouds and RGB images
|   ├── metrics — metrics for evaluating data compression quality
|   └── reduction_methods — different strategies for reducing databases
├── tests — tests of the tool
├── .gitignore
└── README.md
```



