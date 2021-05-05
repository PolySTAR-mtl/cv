import uff
from graphsurgeon import update_node, create_node, create_plugin_node, DynamicGraph
from pathlib import Path
from typing import List, Tuple

from dataclasses import dataclass
from numpy import float32, array
from tensorrt.tensorrt import Logger, init_libnvinfer_plugins, UffParser, Builder

from research.constants import PIPELINES_DIR


@dataclass
class ModelSpec:
    num_classes: int
    input_order: List[int]  # order of loc_data, conf_data, priorbox_data
    input_dim: Tuple[int, int, int] = (3, 300, 300)
    min_size: float = 0.2
    max_size: float = 0.95


def replace_addv2(graph):
    """Replace all 'AddV2' in the graph with 'Add'.

    'AddV2' is not supported by UFF parser.

    Reference:
    1. https://github.com/jkjung-avt/tensorrt_demos/issues/113#issuecomment-629900809
    """
    for node in graph.find_nodes_by_op("AddV2"):
        update_node(node, op="Add")
    return graph


def replace_fusedbnv3(graph):
    """Replace all 'FusedBatchNormV3' in the graph with 'FusedBatchNorm'.

    'FusedBatchNormV3' is not supported by UFF parser.

    Reference:
    1. https://devtalk.nvidia.com/default/topic/1066445/tensorrt/tensorrt-6-0-1-tensorflow-1-14-no-conversion-function-registered-for-layer-fusedbatchnormv3-yet/post/5403567/#5403567
    2. https://github.com/jkjung-avt/tensorrt_demos/issues/76#issuecomment-607879831
    """
    for node in graph.find_nodes_by_op("FusedBatchNormV3"):
        update_node(node, op="FusedBatchNorm")
    return graph


def add_anchor_input(graph):
    """Add the missing const input for the GridAnchor node.

    Reference:
    1. https://www.minds.ai/post/deploying-ssd-mobilenet-v2-on-the-nvidia-jetson-and-nano-platforms
    """
    data = array([1, 1], dtype=float32)
    anchor_input = create_node("AnchorInput", "Const", value=data)
    graph.append(anchor_input)
    graph.find_nodes_by_op("GridAnchor_TRT")[0].input.insert(0, "AnchorInput")
    return graph


def add_plugin(graph: DynamicGraph, spec: ModelSpec):
    """add_plugin

    Reference:
    1. https://github.com/AastaNV/TRT_object_detection/blob/master/config/model_ssd_mobilenet_v1_coco_2018_01_28.py
    2. https://github.com/AastaNV/TRT_object_detection/blob/master/config/model_ssd_mobilenet_v2_coco_2018_03_29.py
    3. https://devtalk.nvidia.com/default/topic/1050465/jetson-nano/how-to-write-config-py-for-converting-ssd-mobilenetv2-to-uff-format/post/5333033/#5333033
    """

    all_assert_nodes = graph.find_nodes_by_op("Assert")
    graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)

    all_identity_nodes = graph.find_nodes_by_op("Identity")
    graph.forward_inputs(all_identity_nodes)

    input = create_plugin_node(name="Input", op="Placeholder", shape=(1,) + spec.input_dim)

    prior_box = create_plugin_node(
        name="MultipleGridAnchorGenerator",
        op="GridAnchor_TRT",
        minSize=spec.min_size,  # was 0.2
        maxSize=spec.max_size,  # was 0.95
        aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
        variance=[0.1, 0.1, 0.2, 0.2],
        featureMapShapes=[19, 10, 5, 3, 2, 1],
        numLayers=6,
    )

    nms = create_plugin_node(
        name="NMS",
        op="NMS_TRT",
        shareLocation=1,
        varianceEncodedInTarget=0,
        backgroundLabelId=0,
        confidenceThreshold=0.3,  # was 1e-8
        nmsThreshold=0.6,
        topK=100,
        keepTopK=100,
        numClasses=spec.num_classes,
        inputOrder=spec.input_order,
        confSigmoid=1,
        isNormalized=1,
    )

    concat_priorbox = create_node("concat_priorbox", op="ConcatV2", axis=2)

    concat_box_loc = create_plugin_node("concat_box_loc", op="FlattenConcat_TRT", axis=1, ignoreBatch=0)
    concat_box_conf = create_plugin_node("concat_box_conf", op="FlattenConcat_TRT", axis=1, ignoreBatch=0)

    namespace_for_removal = [
        "ToFloat",
        "image_tensor",
        "Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3",
    ]
    namespace_plugin_map = {
        "MultipleGridAnchorGenerator": prior_box,
        "Postprocessor": nms,
        "Preprocessor": input,
        "ToFloat": input,
        "Cast": input,  # added for models trained with tf 1.15+
        "image_tensor": input,
        "MultipleGridAnchorGenerator/Concatenate": concat_priorbox,  # for 'ssd_mobilenet_v1_coco'
        "Concatenate": concat_priorbox,  # for other models
        "concat": concat_box_loc,
        "concat_1": concat_box_conf,
    }

    graph.remove(
        graph.find_nodes_by_path(["Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3"]),
        remove_exclusive_dependencies=False,
    )  # for 'ssd_inception_v2_coco'

    graph.collapse_namespaces(namespace_plugin_map)
    graph = replace_addv2(graph)
    graph = replace_fusedbnv3(graph)

    if "image_tensor:0" in graph.find_nodes_by_name("Input")[0].input:
        graph.find_nodes_by_name("Input")[0].input.remove("image_tensor:0")
    if "Input" in graph.find_nodes_by_name("NMS")[0].input:
        graph.find_nodes_by_name("NMS")[0].input.remove("Input")
    # Remove the Squeeze to avoid "Assertion 'isPlugin(layerName)' failed"
    graph.forward_inputs(graph.find_node_inputs_by_name(graph.graph_outputs[0], "Squeeze"))
    if "anchors" in [node.name for node in graph.graph_outputs]:
        graph.remove("anchors", remove_exclusive_dependencies=False)
    if len(graph.find_nodes_by_op("GridAnchor_TRT")[0].input) < 1:
        graph = add_anchor_input(graph)
    if "NMS" not in [node.name for node in graph.graph_outputs]:
        graph.remove(graph.graph_outputs, remove_exclusive_dependencies=False)
        if "NMS" not in [node.name for node in graph.graph_outputs]:
            # We expect 'NMS' to be one of the outputs
            raise RuntimeError("bad graph_outputs")

    return graph


def convert(model_dir: Path, spec: ModelSpec):
    uff_temp_file = str(model_dir / "temp.uff")
    trt_logger = Logger(Logger.INFO)
    init_libnvinfer_plugins(trt_logger, "")
    dynamic_graph = add_plugin(DynamicGraph(str(model_dir / "frozen_inference_graph.pb")), spec)
    uff.from_tensorflow(
        dynamic_graph.as_graph_def(),
        output_nodes=["NMS"],
        output_filename=uff_temp_file,
        text=True,
        debug_mode=False,
    )
    with Builder(trt_logger) as builder, builder.create_network() as network, UffParser() as parser:
        builder.max_workspace_size = 1 << 32
        builder.max_batch_size = 1
        builder.fp16_mode = True

        parser.register_input("Input", spec.input_dim)
        parser.register_output("MarkOutput_0")
        parser.parse(uff_temp_file, network)
        engine = builder.build_cuda_engine(network)

        (model_dir / "trt_model.bin").write_bytes(engine.serialize())


if __name__ == "__main__":
    # [0, 2, 1],
    # 102
    # 012
    # 120
    # 201
    convert(
        PIPELINES_DIR
        / "roco-detection"
        / "210428_001538__ssd_mobilenet_v2__600x600__Twitch2_Train_T470149066_T470150052_T470152289_T470153081_T470158483_T470152730_1891_imgs__20000_steps",
        ModelSpec(num_classes=6, input_order=[0, 2, 1], input_dim=(3, 300, 300)),
    )
