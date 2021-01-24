from dataclasses import dataclass


@dataclass
class Camera:
    horizontal_fov: float
    vertical_fov: float

    pixel_size_m: float
    focal_m: float

    vertical_resolution: int
    horizontal_resolution: int
