from injector import inject

from polystar.communication.cs_link_abc import CSLinkABC
from polystar.dependency_injection import make_injector
from polystar.frame_generators.frames_generator_abc import FrameGeneratorABC
from polystar.target_pipeline.debug_pipeline import DebugTargetPipeline
from polystar.target_pipeline.target_pipeline import NoTargetFoundException
from polystar.utils.fps import FPS
from polystar.view.cv2_results_viewer import CV2ResultViewer


@inject
def demo_pipeline_on_camera(pipeline: DebugTargetPipeline, webcam: FrameGeneratorABC, cs_link: CSLinkABC):
    fps, pipeline_fps = FPS(), FPS()
    with CV2ResultViewer("TensorRT demo") as viewer:
        persistence_last_detection: int = 0
        for image in webcam.generate():
            pipeline_fps.skip()
            try:
                target = pipeline.predict_target(image)
                cs_link.send_target(target)
                persistence_last_detection = 5
            except NoTargetFoundException:
                if persistence_last_detection:
                    persistence_last_detection -= 1
                else:
                    cs_link.send_no_target()
            pipeline_fps.tick(), fps.tick()
            viewer.add_debug_info(pipeline.debug_info_)
            viewer.add_text(f"FPS: {fps:.1f} / {pipeline_fps:.1f}", 10, 10, (0, 0, 0))
            viewer.display()
            fps.skip()
            if viewer.finished:
                return


if __name__ == "__main__":
    make_injector().call_with_injection(demo_pipeline_on_camera)
