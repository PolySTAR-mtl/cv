from polystar.common.view.plt_results_viewer import PltResultViewer
from research.common.datasets_v3.roco.roco_dataset import LazyROCODataset
from research.common.datasets_v3.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo


def visualize_dataset(dataset: LazyROCODataset, n_images: int):
    viewer = PltResultViewer(dataset.name)

    for i, (image, annotation, name) in enumerate(dataset, 1):
        viewer.display_image_with_objects(image, annotation.objects)

        if i == n_images:
            return


if __name__ == "__main__":
    visualize_dataset(ROCODatasetsZoo.DJI_ZOOMED.CENTRAL_CHINA.lazy(), 20)
