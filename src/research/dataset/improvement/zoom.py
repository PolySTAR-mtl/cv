from copy import copy
from dataclasses import dataclass
from itertools import islice
from typing import Iterable, List, Tuple

from polystar.common.models.box import Box
from polystar.common.models.image import Image
from polystar.common.target_pipeline.objects_filters.in_box_filter import InBoxObjectFilter
from polystar.common.view.plt_results_viewer import PltResultViewer
from research.common.datasets.roco.roco_annotation import ROCOAnnotation
from research.common.datasets.roco.zoo.roco_dataset_zoo import ROCODatasetsZoo


def crop_image_annotation(
    image: Image, annotation: ROCOAnnotation, box: Box, min_coverage: float, name: str
) -> Tuple[Image, ROCOAnnotation, str]:
    objects = InBoxObjectFilter(box, min_coverage).filter(annotation.objects)
    objects = [copy(o) for o in objects]
    for obj in objects:
        obj.box = Box.from_positions(
            x1=max(0, obj.box.x1 - box.x1),
            y1=max(0, obj.box.y1 - box.y1),
            x2=min(box.x2, obj.box.x2 - box.x1),
            y2=min(box.y2, obj.box.y2 - box.y1),
        )
    return (
        image[box.y1 : box.y2, box.x1 : box.x2],
        ROCOAnnotation(w=box.w, h=box.h, objects=objects, has_rune=False),
        name,
    )


@dataclass
class Zoomer:
    w: int
    h: int
    max_overlap: float
    min_coverage: float

    def zoom(self, image: Image, annotation: ROCOAnnotation, name: str) -> Iterable[Tuple[Image, ROCOAnnotation, str]]:
        boxes = [obj.box for obj in annotation.objects]
        boxes = self._create_views_covering(boxes, annotation)
        boxes = self._remove_overlapping_boxes(boxes)
        return (
            crop_image_annotation(image, annotation, box, self.min_coverage, name=f"{name}_zoom_{i}")
            for (i, box) in enumerate(boxes, 1)
        )

    def _create_views_covering(self, boxes: List[Box], annotation: ROCOAnnotation) -> List[Box]:
        views: List[Box] = []

        while boxes:
            view, boxes = self._find_new_cluster(boxes)
            view = self._re_frame_box_with_respect_of(view, views, annotation)
            views.append(view)
            boxes = self._remove_covered_boxes(boxes, views)

        return views

    def _find_new_cluster(self, boxes: List[Box]) -> Tuple[Box, List[Box]]:
        cluster: Box = max(boxes, key=lambda b: b.area)
        remaining_boxes: List[Box] = []
        for box in boxes:
            conv_hull = cluster.convex_hull_with(box)
            if conv_hull.w <= self.w and conv_hull.h <= self.h:
                cluster = conv_hull
            else:
                remaining_boxes.append(box)
        return cluster, remaining_boxes

    def _re_frame_box_with_respect_of(self, box: Box, boxes: List[Box], annotation: ROCOAnnotation) -> Box:
        missing_width = self.w - box.w
        missing_height = self.h - box.h

        (
            close_box_on_left,
            close_box_on_top,
            close_box_on_right,
            close_box_on_bottom,
        ) = self._find_directions_of_close_boxes_avoidable(box, boxes, missing_width, missing_height)

        dx = -(missing_width // 2) * (not close_box_on_left) * (1 + close_box_on_right)
        dy = -(missing_height // 2) * (not close_box_on_top) * (1 + close_box_on_bottom)

        x = max(0, min(annotation.w - self.w, box.x1 + dx))
        y = max(0, min(annotation.h - self.h, box.y1 + dy))
        return Box.from_size(x, y, self.w, self.h)

    def _remove_covered_boxes(self, boxes: List[Box], views: List[Box]) -> List[Box]:
        return [b for b in boxes if all(b.area_intersection(v) < b.area * self.min_coverage for v in views)]

    def _remove_overlapping_boxes(self, boxes: List[Box]) -> Iterable[Box]:
        rv: List[Box] = []
        threshold = self.w * self.h * self.max_overlap
        for box in boxes:
            if all(cleared_box.area_intersection(box) <= threshold for cleared_box in rv):
                rv.append(box)
        return rv

    @staticmethod
    def _find_directions_of_close_boxes_avoidable(
        box: Box, boxes: List[Box], missing_width: int, missing_height: int
    ) -> Tuple[bool, bool, bool, bool]:
        left, top, right, bottom = False, False, False, False

        for b in boxes:
            dx, dy = box.distance_among_axis(b, 0), box.distance_among_axis(b, 1)
            if dx <= missing_width // 2 and dy <= missing_height // 2:
                left ^= b.x1 <= box.x1
                right ^= b.x1 >= box.x1
                top ^= b.y1 <= box.y1
                bottom ^= b.y1 >= box.y1

        if top and bottom:
            top, bottom = False, False
        if left and right:
            left, right = False, False

        return left, top, right, bottom


if __name__ == "__main__":
    _zoomer = Zoomer(854, 480, 0.15, 0.5)

    for _img, _annot, _name in islice(ROCODatasetsZoo.DJI.NORTH_CHINA.lazy(), 0, 3):
        _viewer = PltResultViewer(f"img {_name}")

        for (_cropped_image, _cropped_annotation, _cropped_name) in _zoomer.zoom(_img, _annot, _name):
            _viewer.display_image_with_objects(_cropped_image, _cropped_annotation.objects)
