from injector import inject

from polystar.communication.cs_link_abc import CSLinkABC
from polystar.communication.togglabe_cs_link import TogglableCSLink
from polystar.dependency_injection import make_injector
from polystar.frame_generators.frames_generator_abc import FrameGeneratorABC
from polystar.models.image import Image
from polystar.target_pipeline.debug_pipeline import DebugTargetPipeline
from polystar.target_pipeline.target_pipeline import NoTargetFoundException
from polystar.utils.fps import FPS
from polystar.view.cv2_results_viewer import CV2ResultViewer


class CameraPipelineDemo:
    @inject
    def __init__(self, pipeline: DebugTargetPipeline, webcam: FrameGeneratorABC, cs_link: CSLinkABC):
        self.cs_link = TogglableCSLink(cs_link, is_on=False)
        self.webcam = webcam
        self.pipeline = pipeline
        self.fps, self.pipeline_fps = FPS(), FPS()
        self.persistence_last_detection = 0

    def run(self):
        with CV2ResultViewer("TensorRT demo", key_callbacks={" ": self.cs_link.toggle}) as viewer:
            for image in self.webcam.generate():
                self.pipeline_fps.skip()
                self._detect(image)
                self.pipeline_fps.tick(), self.fps.tick()
                self._display(viewer)
                self.fps.skip()
                if viewer.finished:
                    return

    def _detect(self, image: Image):
        try:
            target = self.pipeline.predict_target(image)
            self.cs_link.send_target(target)
            self.persistence_last_detection = 5
        except NoTargetFoundException:
            if self.persistence_last_detection:
                self.persistence_last_detection -= 1
            else:
                self.cs_link.send_no_target()

    def _display(self, viewer: CV2ResultViewer):
        viewer.add_debug_info(self.pipeline.debug_info_)
        viewer.add_text(f"FPS: {self.fps:.1f} / {self.pipeline_fps:.1f}", 10, 10, (0, 0, 0))
        viewer.add_text("Communication: " + ("[ON]" if self.cs_link.is_on else "[OFF]"), 10, 30, (0, 0, 0))
        viewer.display()


if __name__ == "__main__":
    make_injector().get(CameraPipelineDemo).run()
