from contextlib import closing

from injector import inject

from polystar.dependency_injection import make_injector
from polystar.frame_generators.frames_generator_abc import FrameGeneratorABC
from polystar.utils.fps import FPS
from polystar.utils.thread import MyThread
from polystar.view.cv2_results_viewer import CV2ResultViewer


@inject
def display_camera(webcam: FrameGeneratorABC):
    fps = FPS()
    fps_camera = FPS()
    with CV2ResultViewer("Live Camera") as viewer:
        for image in webcam:
            viewer.new(image)
            viewer.add_text(f"FPS: {fps.tick():.1f} / {fps_camera.tick():.1f}", 10, 10, (0, 0, 0))
            viewer.display()
            fps_camera.skip()


if __name__ == "__main__":
    with closing(MyThread):
        make_injector().call_with_injection(display_camera)
