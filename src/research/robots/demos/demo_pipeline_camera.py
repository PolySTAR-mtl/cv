from injector import inject

from polystar.common.dependency_injection import make_injector
from polystar.common.frame_generators.frames_generator_abc import FrameGeneratorABC
from polystar.common.target_pipeline.debug_pipeline import DebugTargetPipeline
from polystar.common.target_pipeline.target_pipeline import NoTargetFoundException
from polystar.common.utils.fps import FPS
from polystar.common.view.cv2_results_viewer import CV2ResultViewer


@inject
def demo_pipeline_on_camera(pipeline: DebugTargetPipeline, webcam: FrameGeneratorABC):
    fps, pipeline_fps = FPS(), FPS()
    with CV2ResultViewer("TensorRT demo") as viewer:
        for image in webcam.generate():
            pipeline_fps.skip()
            try:
                pipeline.predict_target(image)
            except NoTargetFoundException:
                pass
            pipeline_fps.tick(), fps.tick()
            viewer.add_debug_info(pipeline.debug_info_)
            viewer.add_text(f"FPS: {fps:.1f} / {pipeline_fps:.1f}", 10, 10, (0, 0, 0))
            viewer.display()
            fps.skip()
            if viewer.finished:
                return


if __name__ == "__main__":
    make_injector().call_with_injection(demo_pipeline_on_camera)
