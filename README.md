# PlaceRecognitionDB
PlaceRecognitionDB is a tool for creating optimally sized databases (containing the minimum number of frames covering the scene) for place recognition task from RGBD and LiDAR data.

The tool supports several basic methods for creating databases, as well as the DominatingSet method for creating optimal databases.

## Datasets format
To use the tool, your data must be in a specific format.
* Color images in any format.
* Depth images corresponding to color images in 16-bit grayscale format or point clouds in `.pcd` format.
* The trajectory containing one pose in each line in `timestamp (optional) tx ty tz qx qy qz qw` format.

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
in the trajectory file must be the same.

## Usage
Please, check `example.ipynb` with usage example.
