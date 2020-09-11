from polystar.common.view.plt_results_viewer import PltResultViewer
from research.common.datasets.roco.roco_dataset import ROCODataset
from research.common.datasets.roco.zoo.roco_datasets_zoo import ROCODatasetsZoo


def visualize_dataset(dataset: ROCODataset, n_images: int):
    viewer = PltResultViewer(dataset.name)

    for i, (image, annotation) in enumerate(dataset, 1):
        viewer.display_image_with_objects(image, annotation.objects)

        if i == n_images:
            return


if __name__ == "__main__":
    visualize_dataset(ROCODatasetsZoo.DJI_ZOOMED.CentralChina, 20)
