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
import faiss

from tqdm import tqdm
from typing import Optional

from vprdb.core import Database
from vprdb.vpr_systems import CosPlace, NetVLAD, SuperGlue


class GlobalLocalization:
    def __init__(
        self,
        global_extractor: CosPlace | NetVLAD,
        source_db: Database,
        local_matcher: Optional[SuperGlue] = None,
    ):
        self.__global_extractor = global_extractor
        self.__local_matcher = local_matcher
        self.__source_db = source_db

        print("Calculating of global descriptors for source DB")
        self.source_global_descs = self.__global_extractor.get_database_descriptors(
            self.__source_db
        )
        self.faiss_index = faiss.IndexFlatL2(self.source_global_descs.shape[1])
        self.faiss_index.add(self.source_global_descs)
        if self.__local_matcher is not None:
            print("Calculating of local features for source DB")
            self.source_local_features = self.__local_matcher.get_database_features(
                self.__source_db
            )

    def predict(self, query_database: Database, k_closest: int = 1) -> list[int]:
        """
        Predicts query matches
        :param query_database: The database for which the predictions will be calculated
        :param k_closest: Specifies how many predictions for each query the global localization should make.
        If this value is greater than 1, the best match will be chosen with local matcher
        :return: Indexes of frames from the database, corresponding to the query frames
        """
        if k_closest < 1:
            raise ValueError("K closest value can't be below 1")
        elif k_closest > 1 and self.__local_matcher is None:
            raise ValueError(
                "You can't use K closest value > 1 because you don't have SuperGlue local matcher"
            )

        print("Calculating of global descriptors")
        queries_global_descs = self.__global_extractor.get_database_descriptors(
            query_database
        )
        _, global_predictions = self.faiss_index.search(queries_global_descs, k_closest)

        if k_closest == 1:
            return [prediction[0] for prediction in global_predictions]
        else:
            res_predictions = []
            print("Calculating of local features")
            queries_local_descs = self.__local_matcher.get_database_features(
                query_database
            )
            print("Matching of local features")
            for i, query in enumerate(
                tqdm(queries_local_descs, total=len(query_database))
            ):
                global_query_predictions = global_predictions[i]
                filtered_db_features = self.source_local_features[
                    global_query_predictions
                ]
                local_prediction = self.__local_matcher.match_feature(
                    query, filtered_db_features
                )
                res_predictions.append(global_query_predictions[local_prediction])
            return res_predictions
