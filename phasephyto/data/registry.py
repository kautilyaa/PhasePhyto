"""Dataset registry shared by training and evaluation entry points."""

from phasephyto.data.datasets import (
    BananaLeafDiseaseDataset,
    CassavaDataset,
    HistologyDataset,
    PlantDiseaseDataset,
    PlantPathology2021Dataset,
    PollenDataset,
    RiceLeafDiseaseDataset,
    RoCoLeDataset,
    WoodDataset,
)

DATASET_MAP = {
    "plant_disease": PlantDiseaseDataset,
    "cassava": CassavaDataset,
    "plant_pathology_2021": PlantPathology2021Dataset,
    "rocole": RoCoLeDataset,
    "rice_leaf": RiceLeafDiseaseDataset,
    "banana_leaf": BananaLeafDiseaseDataset,
    "histology": HistologyDataset,
    "pollen": PollenDataset,
    "wood": WoodDataset,
}
