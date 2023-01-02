import cv2
import numpy as np
import torch
import torchvision

from tqdm import tqdm

from src.core import Database, memory
from src.cos_place.network import GeoLocalizationNet
from src.providers import ColorImageProvider


@memory.cache
def _get_image_descriptor_with_caching(cos_place, image: ColorImageProvider):
    image_bgr = image.color_image
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    base_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    normalized_img = base_transform(image_rgb)
    normalized_img = normalized_img[None, :]
    descriptor = cos_place.model(normalized_img.to(cos_place.device))
    return descriptor


class CosPlace:
    def __init__(self, backbone: str, fc_output_dim: int, path_to_weights: str):
        self.backbone = backbone
        self.fc_output_dim = fc_output_dim
        self.path_to_weights = path_to_weights

        self.model = GeoLocalizationNet(backbone, fc_output_dim)
        model_state_dict = torch.load(self.path_to_weights)
        self.model.load_state_dict(model_state_dict)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    # This used by Joblib for hashing CosPlace class
    def __getstate__(self):
        state = dict()
        state["backbone"] = self.backbone
        state["fc_output_dim"] = self.fc_output_dim
        state["path_to_weights"] = self.path_to_weights
        return state

    # This used by Joblib for getting metadata about CosPlace class
    def __repr__(self):
        return f"{self.backbone}, {self.fc_output_dim}, {self.path_to_weights}"

    def get_database_descriptors(self, database: Database):
        """
        Gets database RGB images CosPlace descriptors
        :param database: Database for getting descriptors
        :return: Descriptors for database images
        """
        image_providers = database.images
        with torch.no_grad():
            all_descriptors = np.empty(
                (len(image_providers), self.fc_output_dim), dtype="float32"
            )
            for i, image in tqdm(
                enumerate(image_providers), total=len(image_providers)
            ):
                descriptor = _get_image_descriptor_with_caching(self, image)
                descriptor = descriptor.cpu().numpy()
                all_descriptors[i] = descriptor
        return all_descriptors
