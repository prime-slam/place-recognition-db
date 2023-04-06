#  Copyright (c) 2023, Ivan Moskalenko, Anastasiia Kornilova
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""
The `io` submodule allows to read datasets from the hard disk and export them.
## Datasets format
To use the tool, your data must be in a specific format
* Color images in any format
* Depth images in 16-bit grayscale format or point clouds in `.pcd` format.
The depth images must match the color images pixel by pixel (and therefore have the same resolution).
You should also know the intrinsic parameters of the camera and the depth scale if you use depth images
* The trajectory containing one pose in each line in `timestamp tx ty tz qx qy qz qw` format.
Timestamp is an optional argument that is not used in the library

Thus, the tool supports the [TUM](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats) format for datasets.

The correspondence between poses, depth images, and color images
is determined based on the order of lines in the trajectory file and the alphabetical order of the files.

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
"""
from vprdb.io.export_utils import export
from vprdb.io.read_utils import read_dataset_from_depth, read_dataset_from_point_clouds
