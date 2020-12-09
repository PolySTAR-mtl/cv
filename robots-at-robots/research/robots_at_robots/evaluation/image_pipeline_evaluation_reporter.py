from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from math import log
from os.path import relpath
from pathlib import Path
from typing import Dict, Generic, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes, logging
from matplotlib.figure import Figure
from pandas import DataFrame

from polystar.common.pipeline.classification.classification_pipeline import EnumT
from polystar.common.pipeline.pipeline import Pipeline
from polystar.common.utils.dataframe import Format, format_df_row, format_df_rows, make_formater
from polystar.common.utils.markdown import MarkdownFile
from polystar.common.utils.time import create_time_id
from research.common.constants import DSET_DIR, EVALUATION_DIR
from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.robots_at_robots.evaluation.image_pipeline_evaluator import (
    ClassificationResults,
    ImagePipelineEvaluator,
    SetClassificationResults,
)


class Metric(Enum):
    F1_WEIGHTED_AVG = ("f1-score", "weighted avg")
    ACCURACY = ("precision", "accuracy")

    def __str__(self):
        if self.value[1] == "accuracy":
            return "accuracy"
        return " ".join(self.value)

    def __getitem__(self, item):
        return self.value[item]


@dataclass
class ImagePipelineEvaluationReporter(Generic[EnumT]):
    evaluator: ImagePipelineEvaluator[EnumT]
    evaluation_project: str
    experiment_name: str
    main_metric: Metric = Metric.F1_WEIGHTED_AVG
    other_metrics: List[Metric] = field(default_factory=lambda: [Metric.ACCURACY])

    def __post_init__(self):
        self.report_dir = EVALUATION_DIR / self.evaluation_project / f"{create_time_id()}_{self.experiment_name}"

    def report(self, pipelines: Iterable[Pipeline]):
        logging.info(f"Running experiment {self.experiment_name}")

        pipeline2results = self.evaluator.evaluate_pipelines(pipelines)

        with MarkdownFile(self.report_dir / "report.md") as mf:
            mf.title(f"Evaluation report")
            self._report_datasets(mf)
            self._report_aggregated_results(mf, pipeline2results, self.report_dir)
            self._report_pipelines_results(mf, pipeline2results)

            logging.info(f"Report generated at file:///{self.report_dir/'report.md'}")

    def _report_datasets(self, mf: MarkdownFile):
        mf.title("Datasets", level=2)

        mf.title("Training", level=3)
        self._report_dataset(
            mf, self.evaluator.train_roco_datasets, self.evaluator.train_dataset_sizes, self.evaluator.train_labels
        )

        mf.title("Testing", level=3)
        self._report_dataset(
            mf, self.evaluator.test_roco_datasets, self.evaluator.test_dataset_sizes, self.evaluator.test_labels
        )

    @staticmethod
    def _report_dataset(
        mf: MarkdownFile, roco_datasets: List[ROCODatasetBuilder], dataset_sizes: List[int], labels: List[EnumT]
    ):
        total = len(labels)
        labels = [str(label) for label in labels]
        mf.paragraph(f"{total} images")
        df = (
            DataFrame(
                {
                    dataset.name: Counter(labels[start:end])
                    for dataset, start, end in zip(
                        roco_datasets, np.cumsum([0] + dataset_sizes), np.cumsum(dataset_sizes)
                    )
                }
            )
            .fillna(0)
            .sort_index()
        )
        df["Total"] = sum([df[d.name] for d in roco_datasets])
        df["Repartition"] = (df["Total"] / total).map("{:.1%}".format)
        mf.table(df)

    def _report_aggregated_results(
        self, mf: MarkdownFile, pipeline2results: Dict[str, ClassificationResults[EnumT]], report_dir: Path
    ):
        fig_scores, fig_times, aggregated_results = self._aggregate_results(pipeline2results)
        aggregated_scores_image_name = "aggregated_scores.png"
        fig_scores.savefig(report_dir / aggregated_scores_image_name)
        aggregated_times_image_name = "aggregated_times.png"
        fig_times.savefig(report_dir / aggregated_times_image_name)

        mf.title("Aggregated results", level=2)
        mf.image(aggregated_scores_image_name)
        mf.image(aggregated_times_image_name)
        mf.paragraph("On test set:")
        mf.table(aggregated_results[aggregated_results["set"] == "test"].drop(columns="set"))
        mf.paragraph("On train set:")
        mf.table(aggregated_results[aggregated_results["set"] == "train"].drop(columns="set"))

    def _report_pipelines_results(self, mf: MarkdownFile, pipeline2results: Dict[str, ClassificationResults[EnumT]]):
        for pipeline_name, results in sorted(
            pipeline2results.items(),
            key=lambda name_results: name_results[1].test_results.report[self.main_metric[1]][self.main_metric[0]],
            reverse=True,
        ):
            self._report_pipeline_results(mf, pipeline_name, results)

    def _report_pipeline_results(self, mf: MarkdownFile, pipeline_name: str, results: ClassificationResults[EnumT]):
        mf.title(pipeline_name, level=2)

        mf.paragraph(results.full_pipeline_name)

        mf.title("Train results", level=3)
        ImagePipelineEvaluationReporter._report_pipeline_set_results(
            mf, results.train_results, self.evaluator.train_images_paths
        )

        mf.title("Test results", level=3)
        ImagePipelineEvaluationReporter._report_pipeline_set_results(
            mf, results.test_results, self.evaluator.test_images_paths
        )

    @staticmethod
    def _report_pipeline_set_results(
        mf: MarkdownFile, results: SetClassificationResults[EnumT], image_paths: List[Path]
    ):
        mf.title("Metrics", level=4)
        mf.paragraph(f"Inference time: {results.mean_inference_time: .2e} s/img")
        df = DataFrame(results.report)
        format_df_rows(df, ["precision", "recall", "f1-score"], "{:.1%}")
        format_df_row(df, "support", int)
        mf.table(df)
        mf.title("Confusion Matrix:", level=4)
        mf.table(DataFrame(results.confusion_matrix, index=results.unique_labels, columns=results.unique_labels))
        mf.title("25 Mistakes examples", level=4)
        mistakes_idx = np.random.choice(results.mistakes, min(len(results.mistakes), 25), replace=False)
        relative_paths = [
            f"![img]({relpath(str(image_paths[idx]), str(mf.markdown_path.parent))})" for idx in mistakes_idx
        ]
        images_names = [image_paths[idx].relative_to(DSET_DIR) for idx in mistakes_idx]
        mf.table(
            DataFrame(
                {
                    "images": relative_paths,
                    "labels": map(str, results.labels[mistakes_idx]),
                    "predictions": map(str, results.predictions[mistakes_idx]),
                    "image names": images_names,
                }
            ).set_index("images")
        )

    def _aggregate_results(
        self, pipeline2results: Dict[str, ClassificationResults[EnumT]]
    ) -> Tuple[Figure, Figure, DataFrame]:
        sns.set_style()
        sets = ["train", "test"]
        df = DataFrame.from_records(
            [
                {
                    "pipeline": pipeline_name,
                    str(self.main_metric): results.on_set(set_).report[self.main_metric[1]][self.main_metric[0]],
                    "inference time": results.on_set(set_).mean_inference_time,
                    "set": set_,
                }
                for pipeline_name, results in pipeline2results.items()
                # for metric in [self.main_metric]  # + self.other_metrics
                for set_ in sets
            ]
        ).sort_values(["set", str(self.main_metric)], ascending=[True, False])

        return (
            _cat_pipeline_results(df, str(self.main_metric), "{:.1%}", limits=(0, 1)),
            _cat_pipeline_results(df, "inference time", "{:.2e}", log_scale=True),
            df.set_index("pipeline"),
        )


def _cat_pipeline_results(
    df: DataFrame, y: str, fmt: str, limits: Optional[Tuple[float, float]] = None, log_scale: bool = False
) -> Figure:
    grid: sns.FacetGrid = sns.catplot(
        data=df,
        x="pipeline",
        y=y,
        col="set",
        kind="bar",
        sharey=True,
        legend=False,
        col_order=["test", "train"],
        height=10,
    )
    grid.set_xticklabels(rotation=30, ha="right")

    fig: Figure = grid.fig

    _format_axes(fig.get_axes(), fmt, limits=limits, log_scale=log_scale)

    fig.tight_layout()

    fig.suptitle(y)

    return fig


def bar_plot_with_secondary(
    df: DataFrame,
    title: str,
    fmt_y1: Format = str,
    fmt_y2: Format = str,
    y1_log: bool = False,
    y2_log: bool = False,
    limits_y1: Tuple[float, float] = None,
    limits_y2: Tuple[float, float] = None,
    ax: Axes = None,
) -> Tuple[Axes, Axes]:
    if ax is None:
        (_, ax) = plt.subplots()

    y1, y2 = df.columns

    df.plot.bar(rot=0, ax=ax, secondary_y=y2, legend=False, title=title)

    ax1, ax2 = ax, plt.gcf().get_axes()[-1]

    _format_ax(ax1, y1, fmt_y1, y1_log, limits_y1)
    _format_ax(ax2, y2, fmt_y2, y2_log, limits_y2)

    _legend_with_secondary(ax1, ax2)

    return ax1, ax2


def _legend_with_secondary(ax1: Axes, ax2: Axes):
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax1.legend(lines, labels, loc=0)


def _format_axes(axes: List[Axes], fmt: Format, log_scale: bool = False, limits: Optional[Tuple[float, float]] = None):
    for ax in axes:
        _format_ax(ax, fmt, log_scale, limits)


def _format_ax(ax: Axes, fmt: Format, log_scale: bool = False, limits: Optional[Tuple[float, float]] = None):
    if limits:
        ax.set_ylim(*limits)

    if log_scale:
        ax.set_yscale("log")

    m, M = ax.get_ylim()

    fmt = make_formater(fmt)

    for rect in ax.patches:
        h = rect.get_height()
        va = "center"
        if log_scale:
            log_m, log_M = log(m, 10), log(M, 10)
            log_h = log(h, 10)
            if (log_h - log_m) / (log_M - log_m) < 0.25:
                y = h
                va = "bottom"
            else:
                y = pow(10, (log_h + log_m) / 2)
        else:
            if (h - m) / (M - m) < 0.25:
                y = h
                va = "bottom"
            else:
                y = 0.6 * rect.get_height()
        x = rect.get_x() + rect.get_width() / 2
        ax.annotate(
            fmt(rect.get_height()), (x, y), ha="center", va=va, rotation=90,
        )
