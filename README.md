# PlaceRecognitionDB
PlaceRecognitionDB is a tool for creating optimally sized databases (containing the minimum number of frames covering the scene) for place recognition task from RGBD and LiDAR data.

The tool supports several basic methods for creating databases, as well as the DominatingSet method for creating optimal databases.

## Datasets format
To use the tool, your data must be in a specific format
* Color images in any format
* Depth images in 16-bit grayscale format or point clouds in `.pcd` format. The depth images must match the color images 
pixel by pixel (and therefore have the same resolution). You should also know the intrinsic parameters of the camera and the depth scale if you use depth images
* The trajectory containing one pose in each line in `timestamp tx ty tz qx qy qz qw` format. Timestamp is an optional argument that is not used in the library

Thus, the tool supports the [TUM](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats) format for datasets.

The correspondence between poses, depth images, and color images is determined based on the order of lines in the trajectory file and the alphabetical order of the files.

Therefore, the structure of the dataset should look like this:
```
Example dataset
├── color
|   ├── 001.png
|   ├── 002.png
|   ├── ...
├── depth
|   ├── 001.pcd or 001.png
|   ├── 002.pcd or 002.png
|   ├── ...
└── CameraTrajectory.txt
```
The number of color images, depth images (or PCDs) and poses 
in the trajectory file must be the same. The names of folders and the trajectory file are configurable.

## Usage
Please, check `example.ipynb` with usage example.
