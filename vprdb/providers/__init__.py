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
Providers are designed as wrappers over various types of heavy data, to load them into RAM only when needed.
"""
from vprdb.providers.color_image_provider import ColorImageProvider
from vprdb.providers.depth_image_provider import DepthImageProvider
from vprdb.providers.point_cloud_provider import PointCloudProvider
