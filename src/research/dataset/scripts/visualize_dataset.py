from polystar.view.plt_results_viewer import PltResultViewer
from research.common.datasets.roco.roco_dataset import LazyROCODataset
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo


def visualize_dataset(dataset: LazyROCODataset):
    viewer = PltResultViewer(dataset.name)

    for image, annotation, _ in dataset:
        viewer.display_image_with_objects(image, annotation.objects)


if __name__ == "__main__":
    for builder in ROCODatasetsZoo.TWITCH:
        visualize_dataset(builder.cap(5).to_images().build_lazy())
