# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.adapter.base import Adapter

from ax.analysis.analysis import Analysis
from ax.analysis.analysis_card import AnalysisCard
from ax.core.experiment import Experiment
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import override


class Summary(Analysis):
    """
    High-level summary of the Experiment with one row per arm. Any values missing at
    compute time will be represented as None. Columns where every value is None will
    be omitted by default.

    The DataFrame computed will contain one row per arm and the following columns:
        - trial_index: The trial index of the arm
        - arm_name: The name of the arm
        - trial_status: The status of the trial (e.g. RUNNING, SUCCEDED, FAILED)
        - failure_reason: The reason for the failure, if applicable
        - generation_node: The name of the ``GenerationNode`` that generated the arm
        - **METADATA: Any metadata associated with the trial, as specified by the
            Experiment's runner.run_metadata_report_keys field
        - **METRIC_NAME: The observed mean of the metric specified, for each metric
        - **PARAMETER_NAME: The parameter value for the arm, for each parameter
    """

    def __init__(self, omit_empty_columns: bool = True) -> None:
        self.omit_empty_columns = omit_empty_columns

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCard:
        if experiment is None:
            raise UserInputError("`Summary` analysis requires an `Experiment` input")

        return self._create_analysis_card(
            title=(
                "Summary for "
                f"{experiment.name if experiment.has_name else 'Experiment'}"
            ),
            subtitle="High-level summary of the `Trial`-s in this `Experiment`",
            df=experiment.to_df(omit_empty_columns=self.omit_empty_columns),
        )
