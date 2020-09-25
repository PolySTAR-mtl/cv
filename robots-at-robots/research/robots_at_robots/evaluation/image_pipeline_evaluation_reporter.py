from collections import Counter
from dataclasses import dataclass
from math import log
from os.path import relpath
from pathlib import Path
from typing import Any, Dict, Generic, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from pandas import DataFrame

from polystar.common.image_pipeline.image_pipeline import ImagePipeline
from polystar.common.utils.dataframe import Format, format_df_column, format_df_row, format_df_rows, make_formater
from polystar.common.utils.markdown import MarkdownFile
from polystar.common.utils.time import create_time_id
from research.common.constants import DSET_DIR, EVALUATION_DIR
from research.common.datasets.lazy_dataset import TargetT
from research.common.datasets.roco.roco_dataset_builder import ROCODatasetBuilder
from research.robots_at_robots.evaluation.image_pipeline_evaluator import (
    ClassificationResults,
    ImagePipelineEvaluator,
    SetClassificationResults,
)


@dataclass
class ImagePipelineEvaluationReporter(Generic[TargetT]):
    evaluator: ImagePipelineEvaluator[TargetT]
    evaluation_project: str
    main_metric: Tuple[str, str] = ("f1-score", "weighted avg")

    def report(self, pipelines: Iterable[ImagePipeline], evaluation_short_name: str):

        pipeline2results = self.evaluator.evaluate_pipelines(pipelines)

        report_dir = EVALUATION_DIR / self.evaluation_project / f"{evaluation_short_name}_{create_time_id()}"

        with MarkdownFile(report_dir / "report.md") as mf:
            mf.title(f"Evaluation report {evaluation_short_name}")
            self._report_datasets(mf)
            self._report_aggregated_results(mf, pipeline2results, report_dir)
            self._report_pipelines_results(mf, pipeline2results)

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
        mf: MarkdownFile, roco_datasets: List[ROCODatasetBuilder], dataset_sizes: List[int], labels: List[Any]
    ):
        total = len(labels)
        mf.paragraph(f"{total} images")
        df = DataFrame(
            {
                dataset.name: Counter(labels[start:end])
                for dataset, start, end in zip(roco_datasets, np.cumsum([0] + dataset_sizes), np.cumsum(dataset_sizes))
            }
        ).fillna(0)
        df["Total"] = sum([df[d.name] for d in roco_datasets])
        df["Repartition"] = (df["Total"] / total).map("{:.1%}".format)
        mf.table(df)

    def _report_aggregated_results(
        self, mf: MarkdownFile, pipeline2results: Dict[str, ClassificationResults[TargetT]], report_dir: Path
    ):
        fig, (ax_test, ax_train) = plt.subplots(1, 2, figsize=(16, 5))
        aggregated_test_results = self._aggregate_results(pipeline2results, ax_test, "test")
        aggregated_train_results = self._aggregate_results(pipeline2results, ax_train, "train")
        fig.tight_layout()
        aggregated_image_name = "aggregated_test_results.png"
        fig.savefig(report_dir / aggregated_image_name, transparent=True)

        mf.title("Aggregated results", level=2)
        mf.image(aggregated_image_name)
        mf.paragraph("On test set:")
        mf.table(aggregated_test_results)
        mf.paragraph("On train set:")
        mf.table(aggregated_train_results)

    def _report_pipelines_results(self, mf: MarkdownFile, pipeline2results: Dict[str, ClassificationResults[TargetT]]):
        for pipeline_name, results in pipeline2results.items():
            self._report_pipeline_results(mf, pipeline_name, results)

    def _report_pipeline_results(self, mf: MarkdownFile, pipeline_name: str, results: ClassificationResults[TargetT]):
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
        mf: MarkdownFile, results: SetClassificationResults[TargetT], image_paths: List[Path]
    ):
        mf.title("Metrics", level=4)
        mf.paragraph(f"Inference time: {results.mean_inference_time: .2e} s/img")
        df = DataFrame(results.report)
        format_df_rows(df, ["precision", "recall", "f1-score"], "{:.1%}")
        format_df_row(df, "support", int)
        mf.table(df)
        mf.title("Confusion Matrix:", level=4)
        mf.table(DataFrame(results.confusion_matrix, index=results.unique_labels, columns=results.unique_labels))
        mf.title("10 Mistakes examples", level=4)
        mistakes_idx = np.random.choice(results.mistakes, min(len(results.mistakes), 10), replace=False)
        relative_paths = [
            f"![img]({relpath(str(image_paths[idx]), str(mf.markdown_path.parent))})" for idx in mistakes_idx
        ]
        images_names = [image_paths[idx].relative_to(DSET_DIR) for idx in mistakes_idx]
        mf.table(
            DataFrame(
                {
                    "images": relative_paths,
                    "labels": results.labels[mistakes_idx],
                    "predictions": results.predictions[mistakes_idx],
                    "image names": images_names,
                }
            ).set_index("images")
        )

    def _aggregate_results(
        self, pipeline2results: Dict[str, ClassificationResults[TargetT]], ax: Axes, set_: str
    ) -> DataFrame:
        main_metric_name = f"{self.main_metric[0]} {self.main_metric[1]}"
        df = (
            DataFrame.from_records(
                [
                    (
                        pipeline_name,
                        results.on_set(set_).report[self.main_metric[1]][self.main_metric[0]],
                        results.on_set(set_).mean_inference_time,
                    )
                    for pipeline_name, results in pipeline2results.items()
                ],
                columns=["pipeline", main_metric_name, "inf time"],
            )
            .set_index("pipeline")
            .sort_values(main_metric_name, ascending=False)
        )

        bar_plot_with_secondary(df, set_.title(), fmt_y1="{:.1%}", fmt_y2="{:.1e}", y2_log=True, ax=ax)

        format_df_column(df, main_metric_name, "{:.1%}")
        format_df_column(df, "inf time", "{:.2e}")

        return df


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
):
    if ax is None:
        (_, ax) = plt.subplots()

    y1, y2 = df.columns

    df.plot.bar(rot=0, ax=ax, secondary_y=y2, legend=False, title=title)

    ax1, ax2 = ax, plt.gcf().get_axes()[-1]

    _format_ax(ax1, y1, fmt_y1, y1_log, limits_y1)
    _format_ax(ax2, y2, fmt_y2, y2_log, limits_y2)

    _legend_with_secondary(ax1, ax2)


def _legend_with_secondary(ax1: Axes, ax2: Axes):
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax1.legend(lines, labels, loc=0)


def _format_ax(ax: Axes, label: str, fmt: Format, log_scale: bool, limits: Optional[Tuple[float, float]]):
    ax.set_ylabel(label)

    if limits:
        ax.set_ylim(*limits)

    if log_scale:
        ax.set_yscale("log")

    m, _ = ax.get_ylim()

    fmt = make_formater(fmt)

    for p in ax.patches:
        if log_scale:
            h = pow(10, 0.5 * (log(p.get_height(), 10) + log(m, 10)))
        else:
            h = 0.6 * p.get_height()
        ax.annotate(
            fmt(p.get_height()),
            (p.get_x() + p.get_width() / 2.0, h),
            ha="center",
            va="center",
            textcoords="offset points",
        )
