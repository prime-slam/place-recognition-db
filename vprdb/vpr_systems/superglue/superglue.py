#  Copyright (c) 2023, Magic Leap, Ivan Moskalenko, Anastasiia Kornilova
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
#
#  Significant part of our code is based on SuperGluePretrainedNetwork repository
#  (https://github.com/magicleap/SuperGluePretrainedNetwork)
import numpy as np
import torch

from tqdm import tqdm

from vprdb.core import Database
from vprdb.vpr_systems.superglue.models import SuperGlueMatcher, SuperPoint, read_image


class SuperGlue:
    def __init__(
        self,
        path_to_sp_weights,
        path_to_sg_weights,
        resize=(640, 480),
        resize_float=False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Running inference on device "{}"'.format(self.device))

        self.super_point = SuperPoint(path_to_sp_weights).eval().to(self.device)
        self.super_glue_matcher = (
            SuperGlueMatcher(path_to_sg_weights).eval().to(self.device)
        )
        self.resize = resize
        self.resize_float = resize_float

    def get_database_features(self, database: Database):
        """
        Gets database RGB images SuperPoint features
        :param database: Database for getting features
        :return: Features for database images
        """
        features = []
        for image in tqdm(database.color_images):
            inp = read_image(image.path, self.device, self.resize, self.resize_float)
            with torch.no_grad():
                features_for_query = self.super_point({"image": inp})
            features.append(features_for_query)
        return np.asarray(features)

    def match_feature(self, query_feature, db_features):
        """
        Matches query feature with database features
        :param query_feature: Feature for matching
        :param db_features: Database features
        :return: Index of matched image from database
        """
        query_image_results = []
        for db_index, db_feature in enumerate(db_features):
            pred = {k + "0": v for k, v in query_feature.items()}
            pred = {**pred, **{k + "1": v for k, v in db_feature.items()}}
            with torch.no_grad():
                pred = self.super_glue_matcher(pred, self.resize)
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

            matches = pred["matches0"]
            num_matches = np.sum(matches > -1)
            query_image_results.append(num_matches)
        return np.argmax(query_image_results)
