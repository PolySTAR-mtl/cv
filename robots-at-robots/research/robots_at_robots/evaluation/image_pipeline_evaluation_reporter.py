from collections import Counter
from dataclasses import InitVar, dataclass, field
from math import log
from os.path import relpath
from typing import Generic, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes, logging
from matplotlib.figure import Figure
from pandas import DataFrame
from sklearn.metrics import classification_report, confusion_matrix

from polystar.common.pipeline.classification.classification_pipeline import EnumT
from polystar.common.utils.dataframe import Format, format_df_row, format_df_rows, make_formater
from polystar.common.utils.markdown import MarkdownFile
from polystar.common.utils.time import create_time_id
from research.common.constants import DSET_DIR, EVALUATION_DIR
from research.robots_at_robots.evaluation.metrics.accuracy import AccuracyMetric
from research.robots_at_robots.evaluation.metrics.metric_abc import MetricABC
from research.robots_at_robots.evaluation.performance import ClassificationPerformance, ClassificationPerformances
from research.robots_at_robots.evaluation.set import Set


@dataclass
class ImagePipelineEvaluationReporter(Generic[EnumT]):
    evaluation_project: str
    experiment_name: str
    classes: List[EnumT]
    main_metric: MetricABC = field(default_factory=AccuracyMetric)
    other_metrics: InitVar[List[MetricABC]] = None
    _mf: MarkdownFile = field(init=False)
    _performances: ClassificationPerformances = field(init=False)

    def __post_init__(self, other_metrics: List[MetricABC]):
        self.report_dir = EVALUATION_DIR / self.evaluation_project / f"{create_time_id()}_{self.experiment_name}"
        self.all_metrics: List[MetricABC] = [self.main_metric] + (other_metrics or [])

    def report(self, performances: ClassificationPerformances):
        sns.set()
        self._performances = performances
        with MarkdownFile(self.report_dir / "report.md") as self._mf:

            self._mf.title(f"Evaluation report")
            self._report_datasets()
            self._report_aggregated_results()
            self._report_pipelines_results()

            logging.info(f"Report generated at file:///{self.report_dir/'report.md'}")

    def _report_datasets(self):
        self._mf.title("Datasets", level=2)

        self._mf.title("Training", level=3)
        self._report_dataset(self._performances.train)

        self._mf.title("Testing", level=3)
        self._report_dataset(self._performances.test)

    def _report_dataset(self, performances: ClassificationPerformances):
        df = (
            DataFrame({perf.dataset_name: Counter(perf.labels) for perf in performances})
            .fillna(0)
            .sort_index()
            .astype(int)
        )
        df["Total"] = df.sum(axis=1)
        df["Repartition"] = df["Total"] / df["Total"].sum()
        df.loc["Total"] = df.sum()
        df.loc["Repartition"] = df.loc["Total"] / df["Total"]["Total"]
        dset_repartition = df.loc["Repartition"].map("{:.1%}".format)
        df["Repartition"] = df["Repartition"].map("{:.1%}".format)
        df.loc["Repartition"] = dset_repartition
        df.at["Total", "Repartition"] = ""
        df.at["Repartition", "Repartition"] = ""
        df.at["Repartition", "Total"] = ""
        self._mf.table(df)

    def _report_aggregated_results(self):
        fig_scores, fig_times = self._make_aggregate_figures()

        self._mf.title("Aggregated results", level=2)
        self._mf.figure(fig_scores, "aggregated_scores.png")
        self._mf.figure(fig_times, "aggregated_times.png")

        self._mf.paragraph("On test set:")
        self._mf.table(self._make_aggregated_results_for_set(Set.TRAIN))
        self._mf.paragraph("On train set:")
        self._mf.table(self._make_aggregated_results_for_set(Set.TEST))

    def _report_pipelines_results(self):
        for pipeline_name, performances in sorted(
            self._performances.group_by_pipeline().items(),
            key=lambda name_perfs: self.main_metric(name_perfs[1].test.merge()),
            reverse=True,
        ):
            self._report_pipeline_results(pipeline_name, performances)

    def _report_pipeline_results(self, pipeline_name: str, performances: ClassificationPerformances):
        self._mf.title(pipeline_name, level=2)

        self._mf.title("Train results", level=3)
        self._report_pipeline_set_results(performances, Set.TRAIN)

        self._mf.title("Test results", level=3)
        self._report_pipeline_set_results(performances, Set.TEST)

    def _report_pipeline_set_results(self, performances: ClassificationPerformances, set_: Set):
        performances = performances.on_set(set_)
        perf = performances.merge()

        self._mf.title("Metrics", level=4)
        self._report_pipeline_set_metrics(performances, perf, set_)

        self._mf.title("Confusion Matrix:", level=4)
        self._report_pipeline_set_confusion_matrix(perf)

        self._mf.title("25 Mistakes examples", level=4)
        self._report_pipeline_set_mistakes(perf)

    def _report_pipeline_set_metrics(
        self, performances: ClassificationPerformances, perf: ClassificationPerformance, set_: Set
    ):
        fig: Figure = plt.figure(figsize=(9, 6))
        ax: Axes = fig.subplots()
        sns.barplot(
            data=DataFrame(
                [
                    {"dataset": performance.dataset_name, "score": metric(performance), "metric": metric.name}
                    for performance in performances
                    for metric in self.all_metrics
                ]
                + [
                    {"dataset": performance.dataset_name, "score": len(performance) / len(perf), "metric": "support"}
                    for performance in performances
                ]
            ),
            x="dataset",
            hue="metric",
            y="score",
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        pipeline_name = performances.performances[0].pipeline_name
        fig.suptitle(f"{pipeline_name} performance across {set_} datasets")
        _format_ax(ax, "{:.1%}", limits=(0, 1))
        fig.tight_layout()
        self._mf.figure(fig, f"{pipeline_name}_{set_}.png")

        self._mf.paragraph(f"Inference time: {perf.mean_inference_time: .2e} s/img")
        df = DataFrame(classification_report(perf.labels, perf.predictions, output_dict=True))
        format_df_rows(df, ["precision", "recall", "f1-score"], "{:.1%}")
        format_df_row(df, "support", int)
        self._mf.table(df)

    def _report_pipeline_set_confusion_matrix(self, perf: ClassificationPerformance):
        self._mf.table(
            DataFrame(
                confusion_matrix(perf.labels, perf.predictions), index=perf.unique_labels, columns=perf.unique_labels
            )
        )

    def _report_pipeline_set_mistakes(self, perf: ClassificationPerformance):
        mistakes = perf.mistakes
        mistakes_idx = np.random.choice(mistakes, min(len(mistakes), 25), replace=False)
        relative_paths = [
            f"![img]({relpath(str(perf.examples[idx].path), str(self._mf.markdown_path.parent))})"
            for idx in mistakes_idx
        ]
        images_names = [
            f"[{perf.examples[idx].path.relative_to(DSET_DIR)}]"
            f"({relpath(str(perf.examples[idx].path), str(self._mf.markdown_path.parent))})"
            for idx in mistakes_idx
        ]
        self._mf.table(
            DataFrame(
                {
                    "images": relative_paths,
                    "labels": perf.labels[mistakes_idx],
                    "predictions": perf.predictions[mistakes_idx],
                    **{
                        f"p({str(label)})": map("{:.1%}".format, perf.proba[mistakes_idx, i])
                        for i, label in enumerate(self.classes)
                    },
                    "image names": images_names,
                }
            ).set_index("images")
        )

    def _make_aggregate_figures(self) -> Tuple[Figure, Figure]:
        df = DataFrame.from_records(
            [
                {
                    "dataset": perf.dataset_name,
                    "pipeline": perf.pipeline_name,
                    self.main_metric.name: self.main_metric(perf),
                    "time": perf.mean_inference_time,
                    "set": perf.set_.name.lower(),
                    "support": len(perf),
                }
                for perf in self._performances
            ]
        ).sort_values(["set", self.main_metric.name], ascending=[True, False])

        df[f"{self.main_metric.name} "] = list(zip(df[self.main_metric.name], df.support))
        df["time "] = list(zip(df[self.main_metric.name], df.support))

        return (
            _cat_pipeline_results(df, f"{self.main_metric.name} ", "{:.1%}", limits=(0, 1)),
            _cat_pipeline_results(df, "time ", "{:.2e}", log_scale=True),
        )

    def _make_aggregated_results_for_set(self, set_: Set) -> DataFrame:
        pipeline2performances = self._performances.on_set(set_).group_by_pipeline()
        pipeline2performance = {
            pipeline_name: performances.merge() for pipeline_name, performances in pipeline2performances.items()
        }
        return (
            DataFrame(
                [
                    {
                        "pipeline": pipeline_name,
                        self.main_metric.name: self.main_metric(performance),
                        "inference time": performance.mean_inference_time,
                    }
                    for pipeline_name, performance in pipeline2performance.items()
                ]
            )
            .set_index("pipeline")
            .sort_values(self.main_metric.name, ascending=False)
        )


def weighted_mean(x, **kws):
    val, weight = map(np.asarray, zip(*x))
    return (val * weight).sum() / weight.sum()


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
        estimator=weighted_mean,
        orient="v",
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
