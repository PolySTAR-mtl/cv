from typing import Optional

from injector import inject

from polystar.communication.cs_link_abc import CSLinkABC
from polystar.communication.togglabe_cs_link import TogglableCSLink
from polystar.dependency_injection import make_injector
from polystar.frame_generators.frames_generator_abc import FrameGeneratorABC
from polystar.target_pipeline.debug_pipeline import DebugInfo, DebugTargetPipeline
from polystar.target_pipeline.target_abc import SimpleTarget
from polystar.utils.fps import FPS
from polystar.utils.thread import MyThread
from polystar.view.cv2_results_viewer import CV2ResultViewer


class CameraPipelineDemo:
    @inject
    def __init__(self, pipeline: DebugTargetPipeline, webcam: FrameGeneratorABC, cs_link: CSLinkABC):
        self.cs_link = TogglableCSLink(cs_link, is_on=False)
        self.webcam = webcam
        self.pipeline = pipeline
        self.fps = FPS()
        self.persistence_last_detection = 0

    def run(self):
        with CV2ResultViewer("Pipeline demo", key_callbacks={" ": self.cs_link.toggle}) as viewer:
            for debug_info in self.pipeline.flow_debug(self.webcam):
                self._send_target(debug_info.target)
                self._display(viewer, debug_info)

    def _send_target(self, target: Optional[SimpleTarget]):
        if target is not None:
            self.persistence_last_detection = 5
            return self.cs_link.send_target(target)

        if not self.persistence_last_detection:
            return self.cs_link.send_no_target()

        self.persistence_last_detection -= 1

    def _display(self, viewer: CV2ResultViewer, debug_info: DebugInfo):
        viewer.add_debug_info(debug_info)
        viewer.add_text(f"FPS: {self.fps.tick():.1f}", 10, 10, (0, 0, 0))
        viewer.add_text("Communication: " + ("[ON]" if self.cs_link.is_on else "[OFF]"), 10, 30, (0, 0, 0))
        viewer.display()


if __name__ == "__main__":
    make_injector().get(CameraPipelineDemo).run()
    MyThread().stop_all()
