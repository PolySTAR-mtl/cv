from research.common.dataset.roco_dataset import ROCODataset


class UnionDataset(ROCODataset):
    def __init__(self, *datasets: ROCODataset):

        super().__init__(
            image_paths=[image_path for dataset in datasets for image_path in dataset.image_paths],
            annotation_paths=[annotation_path for dataset in datasets for annotation_path in dataset.annotation_paths],
            dataset_name="_".join(dataset.dataset_name for dataset in datasets),
        )
