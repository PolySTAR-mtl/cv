from polystar.common.view.plt_results_viewer import PltResultViewer
from research.common.dataset.dji.dji_roco_zoomed_datasets import DJIROCOZoomedDataset
from research.common.dataset.roco_dataset import ROCODataset


def visualize_dataset(dataset: ROCODataset, n_images: int):
    viewer = PltResultViewer(dataset.dataset_name)

    for i, image in enumerate(dataset.image_annotations, 1):
        viewer.display_image_annotation(image)

        if i == n_images:
            return


if __name__ == "__main__":
    visualize_dataset(DJIROCOZoomedDataset.CentralChina, 10)
