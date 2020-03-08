from polystar.common.models.image import Image
from polystar.common.models.object import Object
from polystar.common.view.plt_display_image_with_annotation import display_image_with_objects


def display_object(image: Image, obj: Object):
    display_image_with_objects(image, [obj])
