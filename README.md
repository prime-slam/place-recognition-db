# PlaceRecognitionDB
[![Lint&Tests](https://github.com/prime-slam/place-recognition-db/actions/workflows/ci.yml/badge.svg)](https://github.com/prime-slam/place-recognition-db/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

PlaceRecognitionDB is a tool for creating optimally sized databases 
(containing the minimum number of frames covering the scene) for place recognition task from RGBD and LiDAR data.

The tool supports several basic methods for creating databases, 
as well as the DominatingSet method for creating optimal databases.

The tool also contains a global Localization pipeline with [CosPlace](https://github.com/gmberton/CosPlace) 
and [NetVLAD](https://github.com/QVPR/Patch-NetVLAD). 
The models of these neural networks can be fine-tuned for a particular previously created database. 
The results of global localization can also be improved with [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork).

We have also developed a set of metrics that can be used to evaluate 
the quality of created databases and the accuracy of VPR systems.

For more, please visit the [PlaceRecognitionDB documentation](https://prime-slam.github.io/place-recognition-db/docs/).
You can also find full information about our research on the [website](https://prime-slam.github.io/place-recognition-db/).

## Datasets format
You can find detailed information about the data format used in the tool [here](https://prime-slam.github.io/place-recognition-db/docs/vprdb/io.html#datasets-format).

## Usage
Please check `example.ipynb` with a small example on creating a database.

## License
This project is licensed under the Apache License — 
see the [LICENSE](https://github.com/prime-slam/place-recognition-db/blob/master/LICENSE) file for details.
