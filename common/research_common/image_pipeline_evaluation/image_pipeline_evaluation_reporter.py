from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Any, Dict, Tuple

import numpy as np
from pandas import DataFrame

from polystar.common.image_pipeline.image_pipeline import ImagePipeline
from polystar.common.utils.dataframe import format_df_rows, format_df_row, format_df_column
from polystar.common.utils.markdown import MarkdownFile
from polystar.common.utils.time import create_time_id
from research_common.constants import EVALUATION_DIR
from research_common.dataset.roco_dataset import ROCODataset
from research_common.image_pipeline_evaluation.image_pipeline_evaluator import (
    ImagePipelineEvaluator,
    ClassificationResults,
    SetClassificationResults,
)


@dataclass
class ImagePipelineEvaluationReporter:
    evaluator: ImagePipelineEvaluator
    evaluation_project: str
    main_metric: Tuple[str, str] = ("f1-score", "weighted avg")

    def report(self, pipelines: Iterable[ImagePipeline], evaluation_short_name: str):

        pipeline2results = self.evaluator.evaluate_pipelines(pipelines)

        with MarkdownFile(
            EVALUATION_DIR / self.evaluation_project / f"{evaluation_short_name}_{create_time_id()}.md"
        ) as mf:
            mf.title(f"Evaluation report {evaluation_short_name}")

            self._report_datasets(mf)

            self._report_aggregated_results(mf, pipeline2results)

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
        mf: MarkdownFile, roco_datasets: List[ROCODataset], dataset_sizes: List[int], labels: List[Any]
    ):
        total = len(labels)
        mf.paragraph(f"{total} images")
        df = DataFrame(
            {
                dataset.dataset_name: Counter(labels[start:end])
                for dataset, start, end in zip(roco_datasets, np.cumsum([0] + dataset_sizes), np.cumsum(dataset_sizes))
            }
        ).fillna(0)
        df["Total"] = sum([df[d.dataset_name] for d in roco_datasets])
        df["Repartition"] = (df["Total"] / total).map("{:.1%}".format)
        mf.table(df)

    def _report_aggregated_results(self, mf: MarkdownFile, pipeline2results: Dict[str, ClassificationResults]):
        aggregated_results = self._aggregate_results(pipeline2results)
        mf.title("Aggregated results", level=2)
        mf.paragraph("On test set:")
        mf.table(aggregated_results)

    def _report_pipelines_results(self, mf: MarkdownFile, pipeline2results: Dict[str, ClassificationResults]):
        for pipeline_name, results in pipeline2results.items():
            self._report_pipeline_results(mf, pipeline_name, results)

    @staticmethod
    def _report_pipeline_results(mf: MarkdownFile, pipeline_name: str, results: ClassificationResults):
        mf.title(pipeline_name, level=2)

        mf.paragraph(results.full_pipeline_name)

        mf.title("Train results", level=3)
        ImagePipelineEvaluationReporter._report_pipeline_set_results(mf, results.train_results)

        mf.title("Test results", level=3)
        ImagePipelineEvaluationReporter._report_pipeline_set_results(mf, results.test_results)

    @staticmethod
    def _report_pipeline_set_results(mf: MarkdownFile, results: SetClassificationResults):
        mf.paragraph(f"Inference time: {results.mean_inference_time: .2e} s/img")
        df = DataFrame(results.report)
        format_df_rows(df, ["precision", "recall", "f1-score"], "{:.1%}")
        format_df_row(df, "support", int)
        mf.table(df)
        mf.paragraph("Confusion Matrix:")
        mf.table(DataFrame(results.confusion_matrix))

    def _aggregate_results(self, pipeline2results: Dict[str, ClassificationResults]) -> DataFrame:
        main_metric_name = f"{self.main_metric[0]} {self.main_metric[1]}"
        df = DataFrame(columns=["pipeline", main_metric_name, "inf time"]).set_index("pipeline")

        for pipeline_name, results in pipeline2results.items():
            df.loc[pipeline_name] = [
                results.test_results.report[self.main_metric[1]][self.main_metric[0]],
                results.test_results.mean_inference_time,
            ]

        df = df.sort_values(main_metric_name, ascending=False)

        format_df_column(df, main_metric_name, "{:.1%}")
        format_df_column(df, "inf time", "{:.2e}")

        return df
