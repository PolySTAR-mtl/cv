from polystar.view.plt_results_viewer import PltResultViewer
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo

if __name__ == "__main__":
    for _img, _annotation, _name in ROCODatasetsZoo.TWITCH.T470149066.to_air().to_images().shuffle().cap(10):
        with PltResultViewer(_name) as _v:
            _v.display_image_with_objects(_img, _annotation.objects)
