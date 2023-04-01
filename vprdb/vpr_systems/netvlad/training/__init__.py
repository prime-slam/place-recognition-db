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
from vprdb.vpr_systems.netvlad.training.pca import pca
from vprdb.vpr_systems.netvlad.training.t_dataset import TDataset
from vprdb.vpr_systems.netvlad.training.tools import save_checkpoint
from vprdb.vpr_systems.netvlad.training.train_epoch import train_epoch
from vprdb.vpr_systems.netvlad.training.validation import validate
