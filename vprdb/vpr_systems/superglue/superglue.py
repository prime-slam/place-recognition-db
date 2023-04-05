# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#                       Ivan Moskalenko
#                       Anastasiia Kornilova
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%
import numpy as np
import torch

from tqdm import tqdm

from vprdb.core import Database
from vprdb.vpr_systems.superglue.models import read_image, SuperGlueMatcher, SuperPoint


class SuperGlue:
    """
    Realisation of [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
    matcher with SuperPoint extractor.
    """

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
