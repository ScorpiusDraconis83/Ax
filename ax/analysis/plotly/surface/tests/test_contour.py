# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.plotly.surface.contour import compute_contour_adhoc, ContourPlot
from ax.core.trial import Trial
from ax.exceptions.core import UserInputError
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import mock_botorch_optimize

from pyre_extensions import assert_is_instance, none_throws


class TestContourPlot(TestCase):
    @mock_botorch_optimize
    def setUp(self) -> None:
        super().setUp()
        self.client = AxClient()
        self.client.create_experiment(
            is_test=True,
            name="foo",
            parameters=[
                {
                    "name": "x",
                    "type": "range",
                    "bounds": [-1.0, 1.0],
                },
                {
                    "name": "y",
                    "type": "range",
                    "bounds": [-1.0, 1.0],
                },
            ],
            objectives={"bar": ObjectiveProperties(minimize=True)},
        )

        for _ in range(10):
            parameterization, trial_index = self.client.get_next_trial()
            self.client.complete_trial(
                trial_index=trial_index,
                raw_data={
                    "bar": parameterization["x"] ** 2 + parameterization["y"] ** 2
                },
            )
        self.expected_subtitle = (
            "The contour plot visualizes the predicted outcomes "
            "for bar across a two-dimensional parameter space, "
            "with other parameters held fixed at their status_quo value "
            "(or mean value if status_quo is unavailable). This plot helps "
            "in identifying regions of optimal performance and understanding "
            "how changes in the selected parameters influence the predicted "
            "outcomes. Contour lines represent levels of constant predicted "
            "values, providing insights into the gradient and potential optima "
            "within the parameter space."
        )
        self.expected_title = "x, y vs. bar"
        self.expected_name = "ContourPlot"
        self.expected_cols = {
            "x",
            "y",
            "bar_mean",
            "bar_sem",
            "sampled",
            "trial_index",
            "arm_name",
        }

    def test_compute(self) -> None:
        analysis = ContourPlot(
            x_parameter_name="x", y_parameter_name="y", metric_name="bar"
        )

        # Test that it fails if no Experiment is provided
        with self.assertRaisesRegex(UserInputError, "requires an Experiment"):
            analysis.compute()
        # Test that it fails if no GenerationStrategy is provided
        with self.assertRaisesRegex(
            UserInputError, "Must provide either a GenerationStrategy or an Adapter"
        ):
            analysis.compute(experiment=self.client.experiment)

        card = analysis.compute(
            experiment=self.client.experiment,
            generation_strategy=self.client.generation_strategy,
        )
        self.assertEqual(
            card.name,
            self.expected_name,
        )
        self.assertEqual(card.title, self.expected_title)
        self.assertEqual(card.subtitle, self.expected_subtitle)
        self.assertEqual(
            {*card.df.columns},
            self.expected_cols,
        )
        self.assertIsNotNone(card.blob)

        # Assert that any row where sampled is True has a value of x that is
        # sampled in at least one trial.
        x_values_sampled = {
            none_throws(assert_is_instance(trial, Trial).arm).parameters["x"]
            for trial in self.client.experiment.trials.values()
        }
        y_values_sampled = {
            none_throws(assert_is_instance(trial, Trial).arm).parameters["y"]
            for trial in self.client.experiment.trials.values()
        }
        self.assertTrue(
            card.df.apply(
                lambda row: row["x"] in x_values_sampled
                and row["y"] in y_values_sampled
                if row["sampled"]
                else True,
                axis=1,
            ).all()
        )

        # Less-than-or-equal to because we may have removed some duplicates
        self.assertTrue(card.df["sampled"].sum() <= len(self.client.experiment.trials))

    def test_compute_adhoc(self) -> None:
        card = compute_contour_adhoc(
            x_parameter_name="x",
            y_parameter_name="y",
            metric_name="bar",
            experiment=self.client.experiment,
            generation_strategy=self.client.generation_strategy,
        )
        self.assertEqual(
            card.name,
            self.expected_name,
        )
        self.assertEqual(card.title, self.expected_title)
        self.assertEqual(card.subtitle, self.expected_subtitle)
        self.assertEqual({*card.df.columns}, self.expected_cols)
        self.assertIsNotNone(card.blob)

        # Assert that any row where sampled is True has a value of x that is
        # sampled in at least one trial.
        x_values_sampled = {
            none_throws(assert_is_instance(trial, Trial).arm).parameters["x"]
            for trial in self.client.experiment.trials.values()
        }
        y_values_sampled = {
            none_throws(assert_is_instance(trial, Trial).arm).parameters["y"]
            for trial in self.client.experiment.trials.values()
        }
        self.assertTrue(
            card.df.apply(
                lambda row: row["x"] in x_values_sampled
                and row["y"] in y_values_sampled
                if row["sampled"]
                else True,
                axis=1,
            ).all()
        )

        # Less-than-or-equal to because we may have removed some duplicates
        self.assertTrue(card.df["sampled"].sum() <= len(self.client.experiment.trials))

    def test_trial_status_filtering(self) -> None:
        trial_index = self.client.experiment.new_trial().index
        self.client.experiment.trials[trial_index].mark_abandoned()

        analysis = ContourPlot(
            x_parameter_name="x", y_parameter_name="y", metric_name="bar"
        )
        card = analysis.compute(
            experiment=self.client.experiment,
            generation_strategy=self.client.generation_strategy,
        )
        self.assertNotIn(
            trial_index,
            card.df["trial_index"].values,
        )
