# PlaceRecognitionDB
PlaceRecognitionDB is a tool for creating optimally sized databases (containing the minimum number of frames covering the scene) for place recognition task from RGBD and LiDAR data.

Several methods of reducing the number of frames in databases have already been developed, as well as some metrics to evaluate the results of compression.

The goal of the project is to create a tool with an optimal compression strategy for any RGBD and LiDAR data.

## Repository structure
```
place-recognition-db
├── src — code of the tool
|   ├── core
|   ├── providers — providers for point clouds and RGB images
|   ├── metrics — metrics for evaluating data compression quality
|   └── reduction_methods — different strategies for reducing databases
├── .gitignore
└── README.md
```



