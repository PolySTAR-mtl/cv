from time import time


class FPS:
    def __init__(self, inertia: float = 0.9):
        self.inertia = inertia
        self.previous_time = time()
        self.fps = 0

    def __str__(self) -> str:
        return str(self.fps)

    def __format__(self, format_spec: str):
        return format(self.fps, format_spec)

    def tick(self) -> float:
        t = time()
        self.fps = self.inertia * self.fps + (1 - self.inertia) / (t - self.previous_time)
        self.previous_time = t
        return self.fps

    def skip(self):
        self.previous_time = time()
