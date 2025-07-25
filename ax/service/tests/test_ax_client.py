#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import sys
import time
from itertools import product
from math import ceil
from typing import TYPE_CHECKING
from unittest import mock
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
from ax.adapter.random import RandomAdapter
from ax.adapter.registry import Cont_X_trans, Generators
from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.outcome_constraint import ObjectiveThreshold, OutcomeConstraint
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.parameter_constraint import OrderConstraint
from ax.core.search_space import HierarchicalSearchSpace
from ax.core.trial import Trial
from ax.core.types import (
    ComparisonOp,
    TEvaluationOutcome,
    TModelPredictArm,
    TParameterization,
    TParamValue,
)
from ax.exceptions.core import (
    DataRequiredError,
    OptimizationComplete,
    UnsupportedError,
    UnsupportedPlotError,
    UserInputError,
)
from ax.exceptions.generation_strategy import MaxParallelismReachedException
from ax.generation_strategy.dispatch_utils import DEFAULT_BAYESIAN_PARALLELISM
from ax.generation_strategy.generation_strategy import (
    GenerationNode,
    GenerationStep,
    GenerationStrategy,
)
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.metrics.branin import branin, BraninMetric
from ax.runners.synthetic import SyntheticRunner
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.best_point import (
    get_best_parameters_from_model_predictions_with_trial_index,
    get_pareto_optimal_parameters,
    observed_pareto,
    predicted_pareto,
)
from ax.service.utils.instantiation import FixedFeatures
from ax.storage.sqa_store.db import init_test_engine_and_session_factory
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.save import save_experiment
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.storage.sqa_store.structs import DBSettings
from ax.utils.common.random import with_rng_seed
from ax.utils.common.testutils import TestCase
from ax.utils.measurement.synthetic_functions import Branin
from ax.utils.testing.core_stubs import (
    DummyEarlyStoppingStrategy,
    get_branin_experiment,
)
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.modeling_stubs import get_observation1, get_observation1trans
from botorch.test_functions.multi_objective import BraninCurrin
from pyre_extensions import assert_is_instance, none_throws

if TYPE_CHECKING:
    from ax.core.types import TTrialEvaluation


RANDOM_SEED = 239
DUMMY_RUN_METADATA = {
    "TEST_KEY": "TEST_VALUE",
    "abc": {123: 456},
}
ARM_NAME = "test_arm_name"


def run_trials_using_recommended_parallelism(
    ax_client: AxClient,
    recommended_parallelism: list[tuple[int, int]],
    total_trials: int,
) -> int:
    remaining_trials = total_trials
    for num_trials, parallelism_setting in recommended_parallelism:
        if num_trials == -1:
            num_trials = remaining_trials
        for _ in range(ceil(num_trials / parallelism_setting)):
            in_flight_trials = []
            if parallelism_setting > remaining_trials:
                parallelism_setting = remaining_trials
            for _ in range(parallelism_setting):
                params, idx = ax_client.get_next_trial()
                in_flight_trials.append((params, idx))
                remaining_trials -= 1
            for _ in range(parallelism_setting):
                params, idx = in_flight_trials.pop()
                # pyre-fixme[6]: For 2nd param expected `Union[List[Tuple[Dict[str, U...
                ax_client.complete_trial(idx, branin(params["x"], params["y"]))
    # If all went well and no errors were raised, remaining_trials should be 0.
    return remaining_trials


def get_branin_currin(minimize: bool = False) -> BraninCurrin:
    return BraninCurrin(negate=not minimize).to(
        dtype=torch.double,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


def get_branin_currin_optimization_with_N_sobol_trials(
    num_trials: int,
    minimize: bool = False,
    include_objective_thresholds: bool = True,
    random_seed: int = RANDOM_SEED,
    outcome_constraints: list[str] | None = None,
) -> tuple[AxClient, BraninCurrin]:
    branin_currin = get_branin_currin(minimize=minimize)
    ax_client = AxClient()
    tracking_metric_names = (
        [elt.split(" ")[0] for elt in outcome_constraints]
        if outcome_constraints is not None
        else None
    )
    ax_client.create_experiment(
        parameters=[
            {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
        ],
        objectives={
            "branin": ObjectiveProperties(
                minimize=minimize,
                # pyre-fixme[6]: For 2nd param expected `Optional[float]` but got
                #  `Optional[Tensor]`.
                threshold=(
                    branin_currin.ref_point[0] if include_objective_thresholds else None
                ),
            ),
            "currin": ObjectiveProperties(
                minimize=minimize,
                # pyre-fixme[6]: For 2nd param expected `Optional[float]` but got
                #  `Optional[Tensor]`.
                threshold=(
                    branin_currin.ref_point[1] if include_objective_thresholds else None
                ),
            ),
        },
        outcome_constraints=outcome_constraints,
        choose_generation_strategy_kwargs={
            "num_initialization_trials": num_trials,
            "random_seed": random_seed,
        },
        tracking_metric_names=tracking_metric_names,
    )
    for _ in range(num_trials):
        parameterization, trial_index = ax_client.get_next_trial()
        x, y = parameterization.get("x"), parameterization.get("y")
        branin = float(branin_currin(torch.tensor([x, y]))[0])
        currin = float(branin_currin(torch.tensor([x, y]))[1])
        raw_data: TTrialEvaluation = {"branin": branin, "currin": currin}
        if tracking_metric_names is not None:
            raw_data["c"] = branin + currin
        ax_client.complete_trial(trial_index, raw_data=raw_data)
    return ax_client, branin_currin


def get_branin_optimization(
    generation_strategy: GenerationStrategy | None = None,
    torch_device: torch.device | None = None,
    support_intermediate_data: bool = False,
) -> AxClient:
    ax_client = AxClient(
        generation_strategy=generation_strategy, torch_device=torch_device
    )
    ax_client.create_experiment(
        name="test_experiment",
        parameters=[
            {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
            {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
        ],
        objectives={"branin": ObjectiveProperties(minimize=True)},
        support_intermediate_data=support_intermediate_data,
    )
    return ax_client


y_values_for_simple_discrete_moo_problem: list[list[float]] = [
    [10.0, 12.0, 11.0],
    [11.0, 10.0, 11.0],
    [12.0, 11.0, 10.0],
]


def get_client_with_simple_discrete_moo_problem(
    minimize: bool,
    use_y0_threshold: bool,
    use_y2_constraint: bool,
) -> AxClient:
    gs = GenerationStrategy(
        steps=[
            GenerationStep(generator=Generators.SOBOL, num_trials=3),
            GenerationStep(
                generator=Generators.BOTORCH_MODULAR,
                num_trials=-1,
                model_kwargs={
                    # To avoid search space exhausted errors.
                    "transforms": Cont_X_trans,
                },
            ),
        ]
    )

    ax_client = AxClient(generation_strategy=gs)
    y1 = ObjectiveProperties(
        minimize=minimize,
        threshold=(-10.5 if minimize else 10.5) if use_y0_threshold else None,
    )
    y2 = ObjectiveProperties(
        minimize=minimize, threshold=0 if use_y0_threshold else None
    )
    outcome_constraint = (
        ["y2 <= -10.5" if minimize else "y2 >= 10.5"] if use_y2_constraint else None
    )
    ax_client.create_experiment(
        name="test_experiment",
        parameters=[
            {
                "name": "x",
                "type": "range",
                # x can only be 0, 1, or 2
                "value_type": "int",
                "bounds": [0, 2],
            }
        ],
        objectives={"y0": y1, "y1": y2},
        tracking_metric_names=["y2"],
        outcome_constraints=outcome_constraint,
    )
    for _ in range(3):
        parameterization, trial_index = ax_client.get_next_trial()
        x = parameterization["x"]
        metrics = y_values_for_simple_discrete_moo_problem[x]
        if minimize:
            metrics = [-m for m in metrics]
        y0, y1, y2 = metrics
        raw_data = {"y0": (y0, 0.0), "y1": (y1, 0.0), "y2": (y2, 0.0)}
        ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
    return ax_client


class TestAxClient(TestCase):
    """Tests service-like API functionality."""

    @mock_botorch_optimize
    def test_interruption(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            objectives={"branin": ObjectiveProperties(minimize=True)},
        )
        for i in range(6):
            parameterization, trial_index = ax_client.get_next_trial()
            self.assertFalse(  # There should be non-complete trials.
                all(t.status.is_terminal for t in ax_client.experiment.trials.values())
            )
            x, y = parameterization.get("x"), parameterization.get("y")
            ax_client.complete_trial(
                trial_index,
                raw_data=assert_is_instance(
                    branin(
                        assert_is_instance(x, float),
                        assert_is_instance(y, float),
                    ),
                    float,
                ),
            )
            old_client = ax_client
            serialized = ax_client.to_json_snapshot()
            ax_client = AxClient.from_json_snapshot(serialized)
            self.assertEqual(len(ax_client.experiment.trials.keys()), i + 1)
            self.assertIsNot(ax_client, old_client)
            self.assertTrue(  # There should be no non-complete trials.
                all(t.status.is_terminal for t in ax_client.experiment.trials.values())
            )

    def test_set_status_quo(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
        )
        self.assertIsNone(ax_client.status_quo)
        status_quo_params = {"x": 1.0, "y": 1.0}
        # pyre-fixme[6]: For 1st param expected `Optional[Dict[str, Union[None,
        #  bool, float, int, str]]]` but got `Dict[str, float]`.
        ax_client.set_status_quo(status_quo_params)
        self.assertEqual(
            ax_client.experiment.status_quo,
            Arm(parameters=status_quo_params, name="status_quo"),
        )

    def test_status_quo_property(self) -> None:
        status_quo_params = {"x": 1.0, "y": 1.0}
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            # pyre-fixme[6]: For 3rd param expected `Optional[Dict[str, Union[None,
            #  bool, float, int, str]]]` but got `Dict[str, float]`.
            status_quo=status_quo_params,
        )
        self.assertEqual(ax_client.status_quo, status_quo_params)
        with self.subTest("it returns a copy"):
            none_throws(ax_client.status_quo).update({"x": 2.0})
            none_throws(ax_client.status_quo)["y"] = 2.0
            self.assertEqual(none_throws(ax_client.status_quo)["x"], 1.0)
            self.assertEqual(none_throws(ax_client.status_quo)["y"], 1.0)

    def test_set_optimization_config_to_moo_with_constraints(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            status_quo={"x": 1.0, "y": 1.0},
        )
        ax_client.set_optimization_config(
            objectives={
                "foo": ObjectiveProperties(minimize=True, threshold=3.1),
                "bar": ObjectiveProperties(minimize=False, threshold=1.0),
            },
            outcome_constraints=["baz >= 7.2%"],
        )
        opt_config = ax_client.experiment.optimization_config
        self.assertEqual(
            # pyre-fixme[16]: `Optional` has no attribute `objective`.
            opt_config.objective.objectives[0].metric.name,
            "foo",
        )
        self.assertEqual(
            opt_config.objective.objectives[0].minimize,
            True,
        )
        self.assertEqual(
            opt_config.objective.objectives[1].metric.name,
            "bar",
        )
        self.assertEqual(
            opt_config.objective.objectives[1].minimize,
            False,
        )
        self.assertEqual(
            # pyre-fixme[16]: `Optional` has no attribute `objective_thresholds`.
            opt_config.objective_thresholds[0],
            ObjectiveThreshold(
                metric=Metric(name="foo", lower_is_better=True),
                bound=3.1,
                relative=False,
                op=ComparisonOp.LEQ,
            ),
        )
        self.assertEqual(
            opt_config.objective_thresholds[1],
            ObjectiveThreshold(
                metric=Metric(name="bar", lower_is_better=False),
                bound=1.0,
                relative=False,
                op=ComparisonOp.GEQ,
            ),
        )
        self.assertEqual(
            # pyre-fixme[16]: `Optional` has no attribute `outcome_constraints`.
            opt_config.outcome_constraints[0],
            OutcomeConstraint(
                metric=Metric(name="baz", lower_is_better=False),
                bound=7.2,
                relative=True,
                op=ComparisonOp.GEQ,
            ),
        )

    def test_set_optimization_config_to_single_objective(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            status_quo={"x": 1.0, "y": 1.0},
        )
        ax_client.set_optimization_config(
            objectives={
                "foo": ObjectiveProperties(minimize=True),
            },
            outcome_constraints=["baz >= 7.2%"],
        )
        opt_config = ax_client.experiment.optimization_config
        self.assertEqual(
            # pyre-fixme[16]: `Optional` has no attribute `objective`.
            opt_config.objective.metric.name,
            "foo",
        )
        self.assertEqual(
            opt_config.objective.minimize,
            True,
        )
        self.assertEqual(
            # pyre-fixme[16]: `Optional` has no attribute `outcome_constraints`.
            opt_config.outcome_constraints[0],
            OutcomeConstraint(
                metric=Metric(name="baz", lower_is_better=False),
                bound=7.2,
                relative=True,
                op=ComparisonOp.GEQ,
            ),
        )

    def test_set_optimization_config_without_objectives_raises_error(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            status_quo={"x": 1.0, "y": 1.0},
        )
        original_opt_config = ax_client.experiment.optimization_config
        with self.assertRaisesRegex(
            ValueError, "optimization config not set because it was missing objectives"
        ):
            ax_client.set_optimization_config(
                outcome_constraints=["baz >= 7.2%"],
            )
        self.assertEqual(original_opt_config, ax_client.experiment.optimization_config)

    @patch(
        "ax.adapter.base.observations_from_data",
        autospec=True,
        return_value=([get_observation1(first_metric_name="branin")]),
    )
    @patch(
        "ax.adapter.random.RandomAdapter.get_training_data",
        autospec=True,
        return_value=([get_observation1(first_metric_name="branin")]),
    )
    @patch(
        "ax.adapter.random.RandomAdapter._predict",
        autospec=True,
        return_value=[get_observation1trans(first_metric_name="branin").data],
    )
    @patch(
        "ax.adapter.random.RandomAdapter.feature_importances",
        autospec=True,
        return_value={"x": 0.9, "y": 1.1},
    )
    @mock_botorch_optimize
    def test_default_generation_strategy_continuous(self, _a, _b, _c, _d) -> None:
        """
        Test that Sobol+BoTorch is used if no GenerationStrategy is provided.
        """
        ax_client = get_branin_optimization()
        self.assertEqual(
            [s.generator for s in none_throws(ax_client.generation_strategy)._steps],
            [Generators.SOBOL, Generators.BOTORCH_MODULAR],
        )
        with self.assertRaisesRegex(ValueError, ".* no trials"):
            ax_client.get_optimization_trace(objective_optimum=branin.fmin)
        for i in range(6):
            gen_limit, opt_complete = ax_client.get_current_trial_generation_limit()
            self.assertFalse(opt_complete)
            if i < 5:
                self.assertEqual(gen_limit, 5 - i)
            else:
                self.assertEqual(gen_limit, DEFAULT_BAYESIAN_PARALLELISM)
            parameterization, trial_index = ax_client.get_next_trial()
            x, y = parameterization.get("x"), parameterization.get("y")
            ax_client.complete_trial(
                trial_index,
                raw_data={
                    "branin": (
                        assert_is_instance(
                            branin(
                                assert_is_instance(x, float),
                                assert_is_instance(y, float),
                            ),
                            float,
                        ),
                        0.0,
                    )
                },
                sample_size=i,
            )
        # pyre-fixme[16]: `Optional` has no attribute `_model_key`.
        self.assertEqual(ax_client.generation_strategy.adapter._model_key, "BoTorch")
        ax_client.get_optimization_trace(objective_optimum=branin.fmin)
        ax_client.get_contour_plot()
        trials_df = ax_client.get_trials_data_frame()
        self.assertIn("x", trials_df)
        self.assertIn("y", trials_df)
        self.assertIn("branin", trials_df)
        self.assertEqual(len(trials_df), 6)

    @patch(
        "ax.adapter.base.observations_from_data",
        autospec=True,
        return_value=([get_observation1(first_metric_name="branin")]),
    )
    @mock_botorch_optimize
    def test_default_generation_strategy_continuous_gen_trials_in_batches(
        self, _
    ) -> None:
        ax_client = get_branin_optimization()
        # All Sobol trials should be able to be generated at once.
        sobol_trials_dict, is_complete = ax_client.get_next_trials(max_trials=10)
        self.assertEqual(len(sobol_trials_dict), 5)
        self.assertFalse(is_complete)
        # Now no trials should be generated since more need completion before GPEI.
        empty_trials_dict, is_complete = ax_client.get_next_trials(max_trials=10)
        self.assertEqual(len(empty_trials_dict), 0)
        self.assertFalse(is_complete)
        for idx, parameterization in sobol_trials_dict.items():
            ax_client.complete_trial(
                idx,
                raw_data={
                    "branin": (
                        assert_is_instance(
                            branin(
                                assert_is_instance(parameterization.get("x"), float),
                                assert_is_instance(parameterization.get("y"), float),
                            ),
                            float,
                        ),
                        0.0,
                    )
                },
            )
        # Now one batch of GPEI trials can be produced, limited by parallelism.
        trials_dict, is_complete = ax_client.get_next_trials(max_trials=10)
        self.assertEqual(len(trials_dict), 3)
        self.assertFalse(is_complete)

    @patch(
        f"{GenerationStrategy.__module__}.GenerationStrategy._gen_with_multiple_nodes",
        side_effect=OptimizationComplete("test error"),
    )
    def test_optimization_complete(self, _mock_gen) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            objectives={"branin": ObjectiveProperties(minimize=True)},
        )
        trials, completed = ax_client.get_next_trials(max_trials=3)
        self.assertEqual(trials, {})
        self.assertTrue(completed)

    def test_sobol_generation_strategy_completion(self) -> None:
        ax_client = get_branin_optimization(
            generation_strategy=GenerationStrategy(
                [GenerationStep(Generators.SOBOL, num_trials=3)]
            )
        )
        # All Sobol trials should be able to be generated at once and optimization
        # should be completed once they are generated.
        sobol_trials_dict, is_complete = ax_client.get_next_trials(max_trials=10)
        self.assertEqual(len(sobol_trials_dict), 3)
        self.assertTrue(is_complete)

        empty_trials_dict, is_complete = ax_client.get_next_trials(max_trials=10)
        self.assertEqual(len(empty_trials_dict), 0)
        self.assertTrue(is_complete)

    def test_save_and_load_generation_strategy(self) -> None:
        init_test_engine_and_session_factory(force_init=True)
        config = SQAConfig()
        encoder = Encoder(config=config)
        decoder = Decoder(config=config)
        db_settings = DBSettings(encoder=encoder, decoder=decoder)
        generation_strategy = GenerationStrategy(
            [GenerationStep(Generators.SOBOL, num_trials=3)]
        )
        ax_client = AxClient(
            db_settings=db_settings, generation_strategy=generation_strategy
        )
        ax_client.create_experiment(
            name="unique_test_experiment",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
        )
        second_client = AxClient(db_settings=db_settings)
        second_client.load_experiment_from_database("unique_test_experiment")
        self.assertEqual(second_client.generation_strategy, generation_strategy)

    def test_save_and_load_no_generation_strategy(self) -> None:
        init_test_engine_and_session_factory(force_init=True)
        config = SQAConfig()
        encoder = Encoder(config=config)
        decoder = Decoder(config=config)
        db_settings = DBSettings(encoder=encoder, decoder=decoder)
        experiment = get_branin_experiment(named=True)
        save_experiment(experiment=experiment, config=config)
        client = AxClient(db_settings=db_settings)
        with self.assertRaisesRegex(
            UserInputError, "choose_generation_strategy_kwargs"
        ):
            client.load_experiment_from_database(experiment.name)

        client = AxClient(db_settings=db_settings)
        client.load_experiment_from_database(
            experiment_name=experiment.name, choose_generation_strategy_kwargs={}
        )
        self.assertIsNotNone(client.generation_strategy)

    @patch(
        f"{AxClient.__module__}.AxClient._save_experiment_to_db_if_possible",
        side_effect=Exception("patched db exception"),
    )
    def test_db_write_failure_on_create_experiment(self, _mock_save_experiment) -> None:
        init_test_engine_and_session_factory(force_init=True)
        config = SQAConfig()
        encoder = Encoder(config=config)
        decoder = Decoder(config=config)
        db_settings = DBSettings(encoder=encoder, decoder=decoder)
        ax_client = AxClient(
            db_settings=db_settings,
        )

        with self.assertRaises(
            expected_exception=Exception, msg="patched db exception"
        ):
            ax_client.create_experiment(
                name="unique_test_experiment1",
                parameters=[
                    {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                    {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
                ],
            )

    @patch(
        "ax.adapter.base.observations_from_data",
        autospec=True,
        return_value=([get_observation1(first_metric_name="branin")]),
    )
    @patch(
        "ax.adapter.random.RandomAdapter.get_training_data",
        autospec=True,
        return_value=([get_observation1(first_metric_name="branin")]),
    )
    @patch(
        "ax.adapter.random.RandomAdapter._predict",
        autospec=True,
        return_value=[get_observation1trans(first_metric_name="branin").data],
    )
    @patch(
        "ax.adapter.random.RandomAdapter.feature_importances",
        autospec=True,
        return_value={"x": 0.9, "y": 1.1},
    )
    @mock_botorch_optimize
    def test_default_generation_strategy_continuous_for_moo(
        self, _a, _b, _c, _d
    ) -> None:
        """Test that Sobol+MOO is used if no GenerationStrategy is provided."""
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            objectives={
                "branin": ObjectiveProperties(minimize=True, threshold=1.0),
                "b": ObjectiveProperties(minimize=True, threshold=1.0),
            },
        )
        self.assertEqual(
            [s.generator for s in none_throws(ax_client.generation_strategy)._steps],
            [Generators.SOBOL, Generators.BOTORCH_MODULAR],
        )
        with self.assertRaisesRegex(ValueError, ".* no trials"):
            ax_client.get_optimization_trace(objective_optimum=branin.fmin)
        for i in range(6):
            with mock.patch("ax.service.ax_client.logger.info") as mock_log:
                parameterization, trial_index = ax_client.get_next_trial()
            log_message = mock_log.call_args.args[0]
            if i < 5:
                expected_model = "Sobol"
            else:
                expected_model = "BoTorch"
            self.assertIn(f"using model {expected_model}", log_message)
            x, y = parameterization.get("x"), parameterization.get("y")
            ax_client.complete_trial(
                trial_index,
                raw_data={
                    "branin": (
                        assert_is_instance(
                            branin(
                                assert_is_instance(x, float),
                                assert_is_instance(y, float),
                            ),
                            float,
                        ),
                        0.0,
                    ),
                    "b": (
                        assert_is_instance(
                            branin(
                                assert_is_instance(x, float),
                                assert_is_instance(y, float),
                            ),
                            float,
                        ),
                        0.0,
                    ),
                },
                sample_size=i,
            )
        # pyre-fixme[16]: `Optional` has no attribute `_model_key`.
        self.assertEqual(ax_client.generation_strategy.adapter._model_key, "BoTorch")
        ax_client.get_contour_plot(metric_name="branin")
        ax_client.get_contour_plot(metric_name="b")
        trials_df = ax_client.get_trials_data_frame()
        self.assertIn("x", trials_df)
        self.assertIn("y", trials_df)
        self.assertIn("branin", trials_df)
        self.assertIn("b", trials_df)
        self.assertEqual(len(trials_df), 6)

        with self.subTest("it raises UnsupportedError for get_optimization_trace"):
            with self.assertRaises(UnsupportedError):
                ax_client.get_optimization_trace(objective_optimum=branin.fmin)

        with self.subTest(
            "it raises UnsupportedError for get_contour_plot without metric"
        ):
            with self.assertRaises(UnsupportedError):
                ax_client.get_contour_plot()

    def test_create_experiment(self) -> None:
        """Test basic experiment creation."""
        ax_client = AxClient(
            GenerationStrategy(
                steps=[GenerationStep(generator=Generators.SOBOL, num_trials=30)]
            )
        )
        with self.assertRaisesRegex(AssertionError, "Experiment not set on Ax client"):
            ax_client.experiment
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {
                    "name": "x",
                    "type": "range",
                    "bounds": [0.001, 0.1],
                    "value_type": "float",
                    "log_scale": True,
                    "digits": 6,
                },
                {
                    "name": "y",
                    "type": "choice",
                    "values": [1, 2, 3],
                    "value_type": "int",
                    "is_ordered": True,
                },
                {"name": "x3", "type": "fixed", "value": 2, "value_type": "int"},
                {
                    "name": "x4",
                    "type": "range",
                    "bounds": [1.0, 3.0],
                    "value_type": "int",
                },
                {
                    "name": "x5",
                    "type": "choice",
                    "values": ["one", "two", "three"],
                    "value_type": "str",
                },
                {
                    "name": "x6",
                    "type": "range",
                    "bounds": [1.0, 3.0],
                    "value_type": "int",
                },
            ],
            objectives={"test_objective": ObjectiveProperties(minimize=True)},
            outcome_constraints=["some_metric >= 3", "some_metric <= 4.0"],
            parameter_constraints=["x4 <= x6"],
            tracking_metric_names=["test_tracking_metric"],
            is_test=True,
        )
        assert ax_client._experiment is not None
        self.assertEqual(ax_client.experiment.__class__.__name__, "Experiment")
        self.assertEqual(ax_client._experiment, ax_client.experiment)
        self.assertEqual(
            # pyre-fixme[16]: `Optional` has no attribute `search_space`.
            ax_client._experiment.search_space.parameters["x"],
            RangeParameter(
                name="x",
                parameter_type=ParameterType.FLOAT,
                lower=0.001,
                upper=0.1,
                log_scale=True,
                digits=6,
            ),
        )
        self.assertEqual(
            ax_client._experiment.search_space.parameters["y"],
            ChoiceParameter(
                name="y",
                parameter_type=ParameterType.INT,
                values=[1, 2, 3],
                is_ordered=True,
            ),
        )
        self.assertEqual(
            ax_client._experiment.search_space.parameters["x3"],
            FixedParameter(name="x3", parameter_type=ParameterType.INT, value=2),
        )
        self.assertEqual(
            ax_client._experiment.search_space.parameters["x4"],
            RangeParameter(
                name="x4", parameter_type=ParameterType.INT, lower=1.0, upper=3.0
            ),
        )
        self.assertEqual(
            ax_client._experiment.search_space.parameters["x5"],
            ChoiceParameter(
                name="x5",
                parameter_type=ParameterType.STRING,
                values=["one", "two", "three"],
            ),
        )
        self.assertEqual(
            # pyre-fixme[16]: `Optional` has no attribute `optimization_config`.
            ax_client._experiment.optimization_config.outcome_constraints[0],
            OutcomeConstraint(
                metric=Metric(name="some_metric", lower_is_better=False),
                op=ComparisonOp.GEQ,
                bound=3.0,
                relative=False,
            ),
        )
        self.assertEqual(
            ax_client._experiment.optimization_config.outcome_constraints[1],
            OutcomeConstraint(
                metric=Metric(name="some_metric", lower_is_better=True),
                op=ComparisonOp.LEQ,
                bound=4.0,
                relative=False,
            ),
        )
        self.assertTrue(ax_client._experiment.optimization_config.objective.minimize)
        self.assertDictEqual(
            # pyre-fixme[16]: `Optional` has no attribute `_tracking_metrics`.
            ax_client._experiment._tracking_metrics,
            {"test_tracking_metric": Metric(name="test_tracking_metric")},
        )
        # pyre-fixme[16]: `Optional` has no attribute
        #  `immutable_search_space_and_opt_config`.
        self.assertTrue(ax_client._experiment.immutable_search_space_and_opt_config)
        self.assertTrue(ax_client.experiment.is_test)

        with self.subTest("objective_name"):
            self.assertEqual(ax_client.objective_name, "test_objective")

        with self.subTest("objective_names"):
            self.assertEqual(ax_client.objective_names, ["test_objective"])

        with self.subTest("metric_names"):
            self.assertEqual(
                ax_client.metric_names,
                {"test_objective", "some_metric", "test_tracking_metric"},
            )

    def test_create_multitype_experiment(self) -> None:
        """
        Test create multitype experiment, add trial type, and add metrics to
        different trial types
        """
        ax_client = AxClient(
            GenerationStrategy(
                steps=[GenerationStep(generator=Generators.SOBOL, num_trials=30)]
            )
        )
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {
                    "name": "x",
                    "type": "range",
                    "bounds": [0.001, 0.1],
                    "value_type": "float",
                    "log_scale": True,
                    "digits": 6,
                },
                {
                    "name": "y",
                    "type": "choice",
                    "values": [1, 2, 3],
                    "value_type": "int",
                    "is_ordered": True,
                },
                {"name": "x3", "type": "fixed", "value": 2, "value_type": "int"},
                {
                    "name": "x4",
                    "type": "range",
                    "bounds": [1.0, 3.0],
                    "value_type": "int",
                },
                {
                    "name": "x5",
                    "type": "choice",
                    "values": ["one", "two", "three"],
                    "value_type": "str",
                },
                {
                    "name": "x6",
                    "type": "range",
                    "bounds": [1.0, 3.0],
                    "value_type": "int",
                },
            ],
            objectives={"test_objective": ObjectiveProperties(minimize=True)},
            outcome_constraints=["some_metric >= 3", "some_metric <= 4.0"],
            parameter_constraints=["x4 <= x6"],
            tracking_metric_names=["test_tracking_metric"],
            is_test=True,
            default_trial_type="test_trial_type",
            default_runner=SyntheticRunner(),
        )

        self.assertEqual(ax_client.experiment.__class__.__name__, "MultiTypeExperiment")
        experiment = assert_is_instance(ax_client.experiment, MultiTypeExperiment)
        self.assertEqual(
            experiment._trial_type_to_runner["test_trial_type"].__class__.__name__,
            "SyntheticRunner",
        )
        self.assertEqual(
            experiment._metric_to_trial_type,
            {
                "test_tracking_metric": "test_trial_type",
                "test_objective": "test_trial_type",
                "some_metric": "test_trial_type",
            },
        )
        experiment.add_trial_type(
            trial_type="test_trial_type_2",
            runner=SyntheticRunner(),
        )
        ax_client.add_tracking_metrics(
            metric_names=[
                "some_metric2_type1",
                "some_metric3_type1",
                "some_metric4_type2",
                "some_metric5_type2",
            ],
            metrics_to_trial_types={
                "some_metric2_type1": "test_trial_type",
                "some_metric4_type2": "test_trial_type_2",
                "some_metric5_type2": "test_trial_type_2",
            },
        )
        self.assertEqual(
            experiment._metric_to_trial_type,
            {
                "test_tracking_metric": "test_trial_type",
                "test_objective": "test_trial_type",
                "some_metric": "test_trial_type",
                "some_metric2_type1": "test_trial_type",
                "some_metric3_type1": "test_trial_type",
                "some_metric4_type2": "test_trial_type_2",
                "some_metric5_type2": "test_trial_type_2",
            },
        )

    def test_create_single_objective_experiment_with_objectives_dict(self) -> None:
        ax_client = AxClient(
            GenerationStrategy(
                steps=[GenerationStep(generator=Generators.SOBOL, num_trials=30)]
            )
        )
        with self.assertRaisesRegex(AssertionError, "Experiment not set on Ax client"):
            ax_client.experiment
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {
                    "name": "x",
                    "type": "range",
                    "bounds": [0.001, 0.1],
                    "value_type": "float",
                    "log_scale": True,
                    "digits": 6,
                },
                {
                    "name": "y",
                    "type": "choice",
                    "values": [1, 2, 3],
                    "value_type": "int",
                    "is_ordered": True,
                },
                {"name": "x3", "type": "fixed", "value": 2, "value_type": "int"},
                {
                    "name": "x4",
                    "type": "range",
                    "bounds": [1.0, 3.0],
                    "value_type": "int",
                },
                {
                    "name": "x5",
                    "type": "choice",
                    "values": ["one", "two", "three"],
                    "value_type": "str",
                },
                {
                    "name": "x6",
                    "type": "range",
                    "bounds": [1.0, 3.0],
                    "value_type": "int",
                },
            ],
            objectives={
                "test_objective": ObjectiveProperties(minimize=True, threshold=2.0),
            },
            outcome_constraints=["some_metric >= 3", "some_metric <= 4.0"],
            parameter_constraints=["x4 <= x6"],
            tracking_metric_names=["test_tracking_metric"],
            is_test=True,
        )
        assert ax_client._experiment is not None
        self.assertEqual(ax_client.objective_name, "test_objective")
        self.assertTrue(ax_client.objective.minimize)

        with self.subTest("objective_name"):
            self.assertEqual(ax_client.objective_name, "test_objective")

        with self.subTest("objective_names"):
            self.assertEqual(ax_client.objective_names, ["test_objective"])

    def test_create_experiment_with_metric_definitions(self) -> None:
        """Test basic experiment creation."""
        ax_client = AxClient()
        with self.assertRaisesRegex(AssertionError, "Experiment not set on Ax client"):
            ax_client.experiment

        metric_definitions = {
            "obj_m1": {"properties": {"m1_opt": "m1_val"}},
            "obj_m2": {"properties": {"m2_opt": "m2_val"}},
            "const_m3": {"properties": {"m3_opt": "m3_val"}},
            "tracking_m4": {"properties": {"m4_opt": "m4_val"}},
        }
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {
                    "name": "x",
                    "type": "range",
                    "bounds": [0.001, 0.1],
                },
            ],
            objectives={
                "obj_m1": ObjectiveProperties(minimize=True, threshold=2.0),
                "obj_m2": ObjectiveProperties(minimize=True, threshold=2.0),
            },
            outcome_constraints=["const_m3 >= 3"],
            tracking_metric_names=["tracking_m4"],
            metric_definitions=metric_definitions,
            is_test=True,
        )
        # pyre-fixme[16]: `Optional` has no attribute `objective`.
        objectives = ax_client.experiment.optimization_config.objective.objectives
        self.assertEqual(objectives[0].metric.name, "obj_m1")
        self.assertEqual(objectives[0].metric.properties, {"m1_opt": "m1_val"})
        self.assertEqual(objectives[1].metric.name, "obj_m2")
        self.assertEqual(objectives[1].metric.properties, {"m2_opt": "m2_val"})
        # pyre-fixme[16]: `Optional` has no attribute `objective_thresholds`.
        thresholds = ax_client.experiment.optimization_config.objective_thresholds
        self.assertEqual(thresholds[0].metric.name, "obj_m1")
        self.assertEqual(thresholds[0].metric.properties, {"m1_opt": "m1_val"})
        self.assertEqual(thresholds[1].metric.name, "obj_m2")
        self.assertEqual(thresholds[1].metric.properties, {"m2_opt": "m2_val"})
        outcome_constraints = (
            # pyre-fixme[16]: `Optional` has no attribute `outcome_constraints`.
            ax_client.experiment.optimization_config.outcome_constraints
        )
        self.assertEqual(outcome_constraints[0].metric.name, "const_m3")
        self.assertEqual(outcome_constraints[0].metric.properties, {"m3_opt": "m3_val"})
        self.assertEqual(ax_client.experiment.tracking_metrics[0].name, "tracking_m4")
        self.assertEqual(
            ax_client.experiment.tracking_metrics[0].properties, {"m4_opt": "m4_val"}
        )
        for k in metric_definitions:
            self.assertEqual(
                ax_client.metric_definitions[k]["properties"],
                metric_definitions[k]["properties"],
            )

    def test_metric_definitions_can_set_a_class(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {
                    "name": "x",
                    "type": "range",
                    "bounds": [0.001, 0.1],
                },
                {
                    "name": "y",
                    "type": "range",
                    "bounds": [0.001, 0.1],
                },
            ],
            is_test=True,
        )
        ax_client.add_tracking_metrics(
            metric_names=["branin"],
            metric_definitions={
                "branin": {
                    "param_names": ["x", "y"],
                    "noise_sd": 0.01,
                    "lower_is_better": False,
                    "metric_class": BraninMetric,
                },
            },
        )
        self.assertEqual(
            ax_client.experiment.metrics["branin"],
            BraninMetric(
                name="branin",
                param_names=["x", "y"],
                noise_sd=0.01,
                lower_is_better=False,
            ),
        )

    def test_set_optimization_config_with_metric_definitions(self) -> None:
        ax_client = AxClient()

        metric_definitions = {
            "obj_m1": {"properties": {"m1_opt": "m1_val"}},
            "obj_m2": {"properties": {"m2_opt": "m2_val"}},
            "const_m3": {"properties": {"m3_opt": "m3_val"}},
            "tracking_m4": {"properties": {"m4_opt": "m4_val"}},
        }
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {
                    "name": "x",
                    "type": "range",
                    "bounds": [0.001, 0.1],
                },
            ],
            is_test=True,
        )
        ax_client.set_optimization_config(
            objectives={
                "obj_m1": ObjectiveProperties(minimize=True, threshold=2.0),
                "obj_m2": ObjectiveProperties(minimize=True, threshold=2.0),
            },
            outcome_constraints=["const_m3 >= 3"],
            metric_definitions=metric_definitions,
        )
        # pyre-fixme[16]: `Optional` has no attribute `objective`.
        objectives = ax_client.experiment.optimization_config.objective.objectives
        self.assertEqual(objectives[0].metric.name, "obj_m1")
        self.assertEqual(objectives[0].metric.properties, {"m1_opt": "m1_val"})
        self.assertEqual(objectives[1].metric.name, "obj_m2")
        self.assertEqual(objectives[1].metric.properties, {"m2_opt": "m2_val"})
        # pyre-fixme[16]: `Optional` has no attribute `objective_thresholds`.
        thresholds = ax_client.experiment.optimization_config.objective_thresholds
        self.assertEqual(thresholds[0].metric.name, "obj_m1")
        self.assertEqual(thresholds[0].metric.properties, {"m1_opt": "m1_val"})
        self.assertEqual(thresholds[1].metric.name, "obj_m2")
        self.assertEqual(thresholds[1].metric.properties, {"m2_opt": "m2_val"})
        outcome_constraints = (
            # pyre-fixme[16]: `Optional` has no attribute `outcome_constraints`.
            ax_client.experiment.optimization_config.outcome_constraints
        )
        self.assertEqual(outcome_constraints[0].metric.name, "const_m3")
        self.assertEqual(outcome_constraints[0].metric.properties, {"m3_opt": "m3_val"})
        self.assertEqual(
            ax_client.metric_definitions["obj_m1"]["properties"],
            metric_definitions["obj_m1"]["properties"],
        )
        self.assertEqual(
            ax_client.metric_definitions["obj_m2"]["properties"],
            metric_definitions["obj_m2"]["properties"],
        )

    def test_add_and_remove_tracking_metrics(self) -> None:
        ax_client = AxClient()

        metric_definitions = {
            "tm1": {"properties": {"m1_opt": "m1_val"}},
        }
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {
                    "name": "x",
                    "type": "range",
                    "bounds": [0.001, 0.1],
                },
            ],
            is_test=True,
        )
        with self.subTest("add tracking metrics with definitions"):
            ax_client.add_tracking_metrics(
                # one with a definition, one without
                metric_names=[
                    "tm1",
                    "tm2",
                ],
                metric_definitions=metric_definitions,
            )
            tracking_metrics = ax_client.experiment.tracking_metrics
            self.assertEqual(len(tracking_metrics), 2)
            self.assertEqual(tracking_metrics[0].name, "tm1")
            self.assertEqual(tracking_metrics[0].properties, {"m1_opt": "m1_val"})
            self.assertEqual(tracking_metrics[1].name, "tm2")
            self.assertEqual(tracking_metrics[1].properties, {})

        with self.subTest("remove tracking metric"):
            ax_client.remove_tracking_metric(metric_name="tm2")
            tracking_metrics = ax_client.experiment.tracking_metrics
            self.assertEqual(len(tracking_metrics), 1)
            self.assertEqual(tracking_metrics[0].name, "tm1")
            self.assertEqual(tracking_metrics[0].properties, {"m1_opt": "m1_val"})

    def test_set_search_space(self) -> None:
        """Test basic experiment creation."""
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {
                    "name": "x1",
                    "type": "range",
                    "bounds": [0.001, 0.1],
                    "value_type": "float",
                    "log_scale": True,
                    "digits": 6,
                },
                {
                    "name": "x2",
                    "type": "range",
                    "bounds": [1.0, 3.0],
                    "value_type": "int",
                },
            ],
            is_test=True,
            immutable_search_space_and_opt_config=False,
        )
        ax_client.set_search_space(
            parameters=[
                {
                    "name": "x1",
                    "type": "range",
                    "bounds": [0.1, 0.2],
                    "value_type": "float",
                    "digits": 6,
                },
                {
                    "name": "x2",
                    "type": "range",
                    "bounds": [1, 2],
                    "value_type": "int",
                },
            ],
            parameter_constraints=["x1 <= x2"],
        )
        param_x1 = RangeParameter(
            name="x1",
            parameter_type=ParameterType.FLOAT,
            lower=0.1,
            upper=0.2,
            log_scale=False,
            digits=6,
        )
        param_x2 = RangeParameter(
            name="x2", parameter_type=ParameterType.INT, lower=1, upper=2
        )
        self.assertEqual(
            ax_client.experiment.search_space.parameters["x1"],
            param_x1,
        )
        self.assertEqual(
            ax_client.experiment.search_space.parameters["x2"],
            param_x2,
        )
        self.assertEqual(
            ax_client.experiment.search_space.parameter_constraints,
            [
                OrderConstraint(
                    lower_parameter=param_x1,
                    upper_parameter=param_x2,
                )
            ],
        )

    def test_create_moo_experiment(self) -> None:
        """Test basic experiment creation."""
        ax_client = AxClient(
            GenerationStrategy(
                steps=[GenerationStep(generator=Generators.SOBOL, num_trials=30)]
            )
        )
        with self.assertRaisesRegex(AssertionError, "Experiment not set on Ax client"):
            ax_client.experiment
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {
                    "name": "x",
                    "type": "range",
                    "bounds": [0.001, 0.1],
                    "value_type": "float",
                    "log_scale": True,
                    "digits": 6,
                },
                {
                    "name": "y",
                    "type": "choice",
                    "values": [1, 2, 3],
                    "value_type": "int",
                    "is_ordered": True,
                },
                {"name": "x3", "type": "fixed", "value": 2, "value_type": "int"},
                {
                    "name": "x4",
                    "type": "range",
                    "bounds": [1.0, 3.0],
                    "value_type": "int",
                },
                {
                    "name": "x5",
                    "type": "choice",
                    "values": ["one", "two", "three"],
                    "value_type": "str",
                },
                {
                    "name": "x6",
                    "type": "range",
                    "bounds": [1.0, 3.0],
                    "value_type": "int",
                },
            ],
            objectives={
                "test_objective_1": ObjectiveProperties(minimize=True, threshold=2.0),
                "test_objective_2": ObjectiveProperties(minimize=False, threshold=7.0),
            },
            outcome_constraints=["some_metric >= 3", "some_metric <= 4.0"],
            parameter_constraints=["x4 <= x6"],
            tracking_metric_names=["test_tracking_metric"],
            is_test=True,
        )
        assert ax_client._experiment is not None
        self.assertEqual(ax_client._experiment, ax_client.experiment)
        self.assertEqual(
            # pyre-fixme[16]: `Optional` has no attribute `search_space`.
            ax_client._experiment.search_space.parameters["x"],
            RangeParameter(
                name="x",
                parameter_type=ParameterType.FLOAT,
                lower=0.001,
                upper=0.1,
                log_scale=True,
                digits=6,
            ),
        )
        self.assertEqual(
            ax_client._experiment.search_space.parameters["y"],
            ChoiceParameter(
                name="y",
                parameter_type=ParameterType.INT,
                values=[1, 2, 3],
                is_ordered=True,
            ),
        )
        self.assertEqual(
            ax_client._experiment.search_space.parameters["x3"],
            FixedParameter(name="x3", parameter_type=ParameterType.INT, value=2),
        )
        self.assertEqual(
            ax_client._experiment.search_space.parameters["x4"],
            RangeParameter(
                name="x4", parameter_type=ParameterType.INT, lower=1.0, upper=3.0
            ),
        )
        self.assertEqual(
            ax_client._experiment.search_space.parameters["x5"],
            ChoiceParameter(
                name="x5",
                parameter_type=ParameterType.STRING,
                values=["one", "two", "three"],
            ),
        )
        # pyre-fixme[16]: `Optional` has no attribute `optimization_config`.
        optimization_config = ax_client._experiment.optimization_config
        self.assertEqual(
            [m.name for m in optimization_config.objective.metrics],
            ["test_objective_1", "test_objective_2"],
        )
        self.assertEqual(
            [o.minimize for o in optimization_config.objective.objectives],
            [True, False],
        )
        self.assertEqual(
            [m.lower_is_better for m in optimization_config.objective.metrics],
            [True, False],
        )
        self.assertEqual(
            [t.metric.name for t in optimization_config.objective_thresholds],
            ["test_objective_1", "test_objective_2"],
        )
        self.assertEqual(
            [t.bound for t in optimization_config.objective_thresholds],
            [2.0, 7.0],
        )
        self.assertEqual(
            [t.op for t in optimization_config.objective_thresholds],
            [ComparisonOp.LEQ, ComparisonOp.GEQ],
        )
        self.assertEqual(
            [t.relative for t in optimization_config.objective_thresholds],
            [False, False],
        )
        self.assertEqual(
            optimization_config.outcome_constraints[0],
            OutcomeConstraint(
                metric=Metric(name="some_metric", lower_is_better=False),
                op=ComparisonOp.GEQ,
                bound=3.0,
                relative=False,
            ),
        )
        self.assertEqual(
            optimization_config.outcome_constraints[1],
            OutcomeConstraint(
                metric=Metric(name="some_metric", lower_is_better=True),
                op=ComparisonOp.LEQ,
                bound=4.0,
                relative=False,
            ),
        )
        self.assertDictEqual(
            # pyre-fixme[16]: `Optional` has no attribute `_tracking_metrics`.
            ax_client._experiment._tracking_metrics,
            {"test_tracking_metric": Metric(name="test_tracking_metric")},
        )
        # pyre-fixme[16]: `Optional` has no attribute
        #  `immutable_search_space_and_opt_config`.
        self.assertTrue(ax_client._experiment.immutable_search_space_and_opt_config)
        self.assertTrue(ax_client.experiment.is_test)

        with self.subTest("objective_name name raises UnsupportedError"):
            with self.assertRaises(UnsupportedError):
                ax_client.objective_name

        with self.subTest("objective_names"):
            self.assertEqual(
                ax_client.objective_names, ["test_objective_1", "test_objective_2"]
            )

    def test_constraint_same_as_objective(self) -> None:
        """Check that we do not allow constraints on the objective metric."""
        ax_client = AxClient(
            GenerationStrategy(
                steps=[GenerationStep(generator=Generators.SOBOL, num_trials=30)]
            )
        )
        with self.assertRaises(ValueError):
            ax_client.create_experiment(
                name="test_experiment",
                parameters=[
                    {"name": "x3", "type": "fixed", "value": 2, "value_type": "int"}
                ],
                objectives={"test_objective": ObjectiveProperties(minimize=True)},
                outcome_constraints=["test_objective >= 3"],
            )

    @mock_botorch_optimize
    def test_raw_data_format(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
        )
        for _ in range(6):
            parameterization, trial_index = ax_client.get_next_trial()
            x, y = parameterization.get("x"), parameterization.get("y")
            # pyre-fixme[6]: For 2nd param expected `Union[List[Tuple[Dict[str, Union...
            ax_client.complete_trial(trial_index, raw_data=(branin(x, y), 0.0))
        with self.assertRaisesRegex(
            UserInputError, "Raw data does not conform to the expected structure."
        ):
            # pyre-fixme[61]: `trial_index` is undefined, or not always defined.
            # pyre-fixme[6]: For 2nd param expected `Union[List[Tuple[Dict[str, Union...
            ax_client.update_trial_data(trial_index, raw_data="invalid_data")

    @mock_botorch_optimize
    def test_raw_data_format_with_map_results(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
            ],
            support_intermediate_data=True,
        )

        for _ in range(6):
            parameterization, trial_index = ax_client.get_next_trial()
            x, y = parameterization.get("x"), parameterization.get("y")
            ax_client.complete_trial(
                trial_index,
                # pyre-fixme[6]: For 2nd param expected `Union[List[Tuple[Dict[str, U...
                raw_data=[
                    ({"y": y / 2.0}, {"objective": (branin(x, y / 2.0), 0.0)}),
                    ({"y": y}, {"objective": (branin(x, y), 0.0)}),
                ],
            )

    def test_keep_generating_without_data(self) -> None:
        # Check that normally number of arms to generate is enforced.
        ax_client = get_branin_optimization()
        for _ in range(5):
            parameterization, trial_index = ax_client.get_next_trial()
        with self.assertRaisesRegex(DataRequiredError, "All trials for current node"):
            ax_client.get_next_trial()
        # Check that with enforce_sequential_optimization off, we can keep
        # generating.
        ax_client = AxClient(enforce_sequential_optimization=False)
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
        )
        self.assertFalse(
            ax_client.generation_strategy._steps[0].enforce_num_trials, False
        )
        self.assertFalse(ax_client.generation_strategy._steps[1].max_parallelism, None)
        for _ in range(10):
            parameterization, trial_index = ax_client.get_next_trial()

    def test_update_running_trial_with_intermediate_data(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
            ],
            objectives={"branin": ObjectiveProperties(minimize=True)},
            support_intermediate_data=True,
        )
        parameterization, trial_index = ax_client.get_next_trial()
        # Launch Trial and update it 3 times with additional data.
        for t in range(3):
            x, y = parameterization.get("x"), parameterization.get("y")
            if t < 2:
                ax_client.update_running_trial_with_intermediate_data(
                    0,
                    # pyre-fixme[6]: For 2nd argument expected `Union[floating[typing...
                    raw_data=[({"t": t}, {"branin": (branin(x, y) + t, 0.0)})],
                )
            if t == 2:
                ax_client.complete_trial(
                    0,
                    # pyre-fixme[6]: For 2nd argument expected `Union[floating[typing...
                    raw_data=[({"t": t}, {"branin": (branin(x, y) + t, 0.0)})],
                )
            # pyre-fixme[16]: `Data` has no attribute `map_df`.
            fetch_data = ax_client.experiment.fetch_data().map_df
            self.assertEqual(len(fetch_data), 0 if t < 2 else 3)
            lookup_data = ax_client.experiment.lookup_data().map_df
            self.assertEqual(len(lookup_data), t + 1)

        no_intermediate_data_ax_client = AxClient()
        no_intermediate_data_ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
            ],
            support_intermediate_data=False,
        )
        parameterization, trial_index = no_intermediate_data_ax_client.get_next_trial()
        with self.assertRaises(ValueError):
            no_intermediate_data_ax_client.update_running_trial_with_intermediate_data(
                0,
                raw_data=[
                    # pyre-fixme[61]: `t` is undefined, or not always defined.
                    ({"t": p_t}, {"branin": (branin(x, y) + t, 0.0)})
                    # pyre-fixme[61]: `t` is undefined, or not always defined.
                    for p_t in range(t + 1)
                ],
            )

    # pyre-fixme[56]: Pyre was not able to infer the type of argument `f"{ax.service....
    @patch(
        f"{get_best_parameters_from_model_predictions_with_trial_index.__module__}"
        + ".get_best_parameters_from_model_predictions_with_trial_index",
        wraps=get_best_parameters_from_model_predictions_with_trial_index,
    )
    def test_get_best_point_no_model_predictions(
        self,
        mock_get_best_parameters_from_model_predictions_with_trial_index: Mock,
    ) -> None:
        ax_client = get_branin_optimization()
        params, idx = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=idx, raw_data={"branin": (0, 0.0)})
        best_idx, best_params, _ = none_throws(ax_client.get_best_trial())
        self.assertEqual(best_idx, idx)
        self.assertEqual(best_params, params)
        self.assertEqual(none_throws(ax_client.get_best_parameters())[0], params)
        mock_get_best_parameters_from_model_predictions_with_trial_index.assert_called()
        mock_get_best_parameters_from_model_predictions_with_trial_index.reset_mock()
        ax_client.get_best_parameters(use_model_predictions=False)
        mock_get_best_parameters_from_model_predictions_with_trial_index.assert_not_called()  # noqa

    def test_trial_completion(self) -> None:
        ax_client = get_branin_optimization()
        params, idx = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=idx, raw_data={"branin": (0, 0.0)})
        metrics_in_data = ax_client.experiment.fetch_data().df["metric_name"].values
        self.assertNotIn("m1", metrics_in_data)
        self.assertIn("branin", metrics_in_data)
        # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
        self.assertEqual(ax_client.get_best_parameters()[0], params)
        params2, idy = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=idy, raw_data=(-1, 0.0))
        self.assertEqual(ax_client.get_best_parameters()[0], params2)
        params3, idx3 = ax_client.get_next_trial()
        ax_client.complete_trial(
            trial_index=idx3, raw_data=-2, metadata={"dummy": "test"}
        )
        self.assertEqual(ax_client.get_best_parameters()[0], params3)
        self.assertEqual(
            # pyre-fixme[16]: `Optional` has no attribute `run_metadata`.
            ax_client.experiment.trials.get(2).run_metadata.get("dummy"),
            "test",
        )
        best_trial_values = ax_client.get_best_parameters()[1]
        self.assertEqual(best_trial_values[0], {"branin": -2.0})
        self.assertTrue(math.isnan(best_trial_values[1]["branin"]["branin"]))

    def test_update_trial_data(self) -> None:
        ax_client = get_branin_optimization(support_intermediate_data=True)
        params, idx = ax_client.get_next_trial()
        # Can't update before completing.
        with self.assertRaisesRegex(ValueError, ".* not in a terminal state"):
            ax_client.update_trial_data(
                trial_index=idx, raw_data=[({"t": 0}, {"branin": (0, 0.0)})]
            )
        ax_client.complete_trial(
            trial_index=idx, raw_data=[({"t": 0}, {"branin": (0, 0.0)})]
        )
        # Cannot complete a trial twice, should use `update_trial_data`.
        with self.assertRaisesRegex(UnsupportedError, ".* already been completed"):
            ax_client.complete_trial(
                trial_index=idx, raw_data=[({"t": 0}, {"branin": (0, 0.0)})]
            )
        # Check that the update changes the data.
        ax_client.update_trial_data(
            trial_index=idx, raw_data=[({"t": 0}, {"branin": (1, 0.0)})]
        )
        df = ax_client.experiment.lookup_data_for_trial(idx)[0].df
        self.assertEqual(len(df), 1)
        self.assertEqual(df["mean"].item(), 1.0)
        self.assertTrue(ax_client.get_trial(idx).status.is_completed)

        # With early stopped trial.
        params, idx = ax_client.get_next_trial()
        ax_client.update_running_trial_with_intermediate_data(
            idx,
            # pyre-fixme[6]: For 2nd argument expected `Union[floating[typing...
            raw_data=[({"t": 0}, {"branin": (branin(*params.values()), 0.0)})],
        )

        ax_client.stop_trial_early(trial_index=idx)
        df = ax_client.experiment.lookup_data_for_trial(idx)[0].df
        self.assertEqual(len(df), 1)
        ax_client.update_trial_data(
            trial_index=idx, raw_data=[({"t": 0}, {"branin": (2, 0.0)})]
        )
        df = ax_client.experiment.lookup_data_for_trial(idx)[0].df
        self.assertEqual(len(df), 1)
        self.assertEqual(df["mean"].item(), 2.0)
        self.assertTrue(ax_client.get_trial(idx).status.is_early_stopped)

        # Failed trial.
        params, idx = ax_client.get_next_trial()
        ax_client.log_trial_failure(trial_index=idx)
        ax_client.update_trial_data(
            trial_index=idx, raw_data=[({"t": 0}, {"branin": (3, 0.0)})]
        )
        df = ax_client.experiment.lookup_data_for_trial(idx)[0].df
        self.assertEqual(df["mean"].item(), 3.0)

        # Incomplete trial fails
        params, idx = ax_client.get_next_trial()
        ax_client.complete_trial(
            trial_index=idx, raw_data=[({"t": 0}, {"missing_metric": (0, 0.0)})]
        )
        self.assertTrue(ax_client.get_trial(idx).status.is_failed)

    def test_incomplete_multi_fidelity_trial(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
            ],
            objectives={"branin": ObjectiveProperties(minimize=True)},
            support_intermediate_data=True,
        )
        # Trial with complete data
        params, idx = ax_client.get_next_trial()
        ax_client.complete_trial(
            trial_index=idx, raw_data=[({"fidelity": 1}, {"branin": (123, 0.0)})]
        )
        self.assertTrue(ax_client.get_trial(idx).status.is_completed)
        # Trial with incomplete data
        params, idx = ax_client.get_next_trial()
        ax_client.complete_trial(
            trial_index=idx,
            raw_data=[({"fidelity": 2}, {"missing_metric": (456, 0.0)})],
        )
        self.assertTrue(ax_client.get_trial(idx).status.is_failed)

    def test_trial_completion_with_metadata_with_iso_times(self) -> None:
        ax_client = get_branin_optimization()
        params, idx = ax_client.get_next_trial()
        ax_client.complete_trial(
            trial_index=idx,
            raw_data={"branin": (0, 0.0)},
            metadata={
                "start_time": "2020-01-01",
                "end_time": "2020-01-05 00:00:00",
            },
        )
        with patch.object(
            RandomAdapter, "_fit", autospec=True, side_effect=RandomAdapter._fit
        ) as mock_fit:
            ax_client.get_next_trial()
            self.assertEqual(mock_fit.call_count, 1)
            features = mock_fit.call_args_list[0][1]["observations"][0].features
            # we're asserting it's actually created real Timestamp objects
            # for the observation features
            self.assertEqual(features.start_time.day, 1)
            self.assertEqual(features.end_time.day, 5)

    def test_trial_completion_with_metadata_millisecond_times(self) -> None:
        ax_client = get_branin_optimization()
        params, idx = ax_client.get_next_trial()
        ax_client.complete_trial(
            trial_index=idx,
            raw_data={"branin": (0, 0.0)},
            metadata={
                "start_time": int(pd.Timestamp("2020-01-01").timestamp() * 1000),
                "end_time": int(pd.Timestamp("2020-01-05").timestamp() * 1000),
            },
        )
        with patch.object(
            RandomAdapter, "_fit", autospec=True, side_effect=RandomAdapter._fit
        ) as mock_fit:
            ax_client.get_next_trial()
            self.assertEqual(mock_fit.call_count, 1)
            features = mock_fit.call_args_list[0][1]["observations"][0].features
            # we're asserting it's actually created real Timestamp objects
            # for the observation features
            self.assertEqual(features.start_time.day, 1)
            self.assertEqual(features.end_time.day, 5)

    def test_abandon_trial(self) -> None:
        ax_client = get_branin_optimization()

        # An abandoned trial adds no data.
        params, idx = ax_client.get_next_trial()
        ax_client.abandon_trial(trial_index=idx)
        data = ax_client.experiment.fetch_data()
        self.assertEqual(len(data.df.index), 0)

        # Can't update a completed trial.
        params2, idx2 = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=idx2, raw_data={"branin": (0, 0.0)})
        with self.assertRaisesRegex(ValueError, ".* in a terminal state."):
            ax_client.abandon_trial(trial_index=idx2)

    def test_ttl_trial(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
        )

        # A ttl trial that ends adds no data.
        params, idx = ax_client.get_next_trial(ttl_seconds=1)
        # pyre-fixme[16]: `Optional` has no attribute `status`.
        self.assertTrue(ax_client.experiment.trials.get(idx).status.is_running)
        time.sleep(1)  # Wait for TTL to elapse.
        self.assertTrue(ax_client.experiment.trials.get(idx).status.is_failed)
        # Also make sure we can no longer complete the trial as it is failed.
        with self.assertRaisesRegex(
            UnsupportedError, ".* has been marked FAILED, so it no longer expects data."
        ):
            ax_client.complete_trial(trial_index=idx, raw_data={"objective": (0, 0.0)})

        params2, idy = ax_client.get_next_trial(ttl_seconds=1)
        ax_client.complete_trial(trial_index=idy, raw_data=(-1, 0.0))
        # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
        self.assertEqual(ax_client.get_best_parameters()[0], params2)

    def test_start_and_end_time_in_trial_completion(self) -> None:
        start_time = pd.Timestamp.now().isoformat()
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
        )
        params, idx = ax_client.get_next_trial()
        ax_client.complete_trial(
            trial_index=idx,
            raw_data=1.0,
            metadata={
                "start_time": start_time,
                "end_time": pd.Timestamp.now().isoformat(),
            },
        )
        dat = ax_client.experiment.fetch_data().df
        self.assertGreater(dat["end_time"][0], dat["start_time"][0])

    def test_fail_on_batch(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
        )
        batch_trial = ax_client.experiment.new_batch_trial(
            generator_run=GeneratorRun(
                arms=[
                    Arm(parameters={"x": 0, "y": 1}),
                    Arm(parameters={"x": 0, "y": 1}),
                ]
            )
        )
        with self.assertRaisesRegex(
            TypeError,
            r"obj is not an instance of cls: obj=.*BatchTrial.*"
            r"cls=<class 'ax.core.trial.Trial'>",
        ):
            ax_client.complete_trial(batch_trial.index, 0)

    def test_log_failure(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
        )
        _, idx = ax_client.get_next_trial()
        ax_client.log_trial_failure(idx, metadata={"dummy": "test"})
        # pyre-fixme[16]: `Optional` has no attribute `status`.
        self.assertTrue(ax_client.experiment.trials.get(idx).status.is_failed)
        self.assertEqual(
            # pyre-fixme[16]: `Optional` has no attribute `run_metadata`.
            ax_client.experiment.trials.get(idx).run_metadata.get("dummy"),
            "test",
        )
        with self.assertRaisesRegex(UnsupportedError, ".* no longer expects"):
            ax_client.complete_trial(idx, {})

    def test_attach_trial_and_get_trial_parameters(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
        )
        params, idx = ax_client.attach_trial(
            parameters={"x": 0.0, "y": 1.0},
            run_metadata=DUMMY_RUN_METADATA,
            arm_name=ARM_NAME,
        )
        ax_client.complete_trial(trial_index=idx, raw_data=5)
        # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
        self.assertEqual(ax_client.get_best_parameters()[0], params)
        self.assertEqual(
            ax_client.get_trial_parameters(trial_index=idx), {"x": 0, "y": 1}
        )
        self.assertEqual(
            none_throws(ax_client.get_trial(trial_index=idx).arm).name, ARM_NAME
        )
        with self.assertRaises(KeyError):
            ax_client.get_trial_parameters(
                trial_index=10
            )  # No trial #10 in experiment.
        with self.assertRaisesRegex(UnsupportedError, ".* is of type"):
            ax_client.attach_trial({"x": 1, "y": 2})

    def test_attach_trial_ttl_seconds(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
        )
        params, idx = ax_client.attach_trial(
            parameters={"x": 0.0, "y": 1.0}, ttl_seconds=1
        )
        # pyre-fixme[16]: `Optional` has no attribute `status`.
        self.assertTrue(ax_client.experiment.trials.get(idx).status.is_running)
        time.sleep(1)  # Wait for TTL to elapse.
        self.assertTrue(ax_client.experiment.trials.get(idx).status.is_failed)
        # Also make sure we can no longer complete the trial as it is failed.
        with self.assertRaisesRegex(
            UnsupportedError, ".* has been marked FAILED, so it no longer expects data."
        ):
            ax_client.complete_trial(trial_index=idx, raw_data=5)

        params2, idx2 = ax_client.attach_trial(
            parameters={"x": 0.0, "y": 1.0}, ttl_seconds=1
        )
        ax_client.complete_trial(trial_index=idx2, raw_data=5)
        # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
        self.assertEqual(ax_client.get_best_parameters()[0], params2)
        self.assertEqual(
            ax_client.get_trial_parameters(trial_index=idx2), {"x": 0, "y": 1}
        )

    def test_attach_trial_numpy(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
        )
        params, idx = ax_client.attach_trial(parameters={"x": 0.0, "y": 1.0})
        ax_client.complete_trial(trial_index=idx, raw_data=np.int32(5))
        # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
        self.assertEqual(ax_client.get_best_parameters()[0], params)

    def test_relative_oc_without_sq(self) -> None:
        """Must specify status quo to have relative outcome constraint."""
        ax_client = AxClient()
        with self.assertRaises(ValueError):
            ax_client.create_experiment(
                name="test_experiment",
                parameters=[
                    {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                    {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
                ],
                objectives={"test_objective": ObjectiveProperties(minimize=True)},
                outcome_constraints=["some_metric <= 4.0%"],
            )

    @mock_botorch_optimize
    def test_recommended_parallelism(self) -> None:
        ax_client = AxClient()
        with self.assertRaisesRegex(AssertionError, "No generation strategy"):
            ax_client.get_max_parallelism()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
        )
        self.assertEqual(ax_client.get_max_parallelism(), [(5, 5), (-1, 3)])
        self.assertEqual(
            run_trials_using_recommended_parallelism(
                ax_client, ax_client.get_max_parallelism(), 20
            ),
            0,
        )
        # With incorrect parallelism setting, the 'need more data' error should
        # still be raised.
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
        )
        with self.assertRaisesRegex(DataRequiredError, "All trials for current node"):
            run_trials_using_recommended_parallelism(ax_client, [(6, 6), (-1, 3)], 20)

    @patch.dict(sys.modules, {"ax.storage.sqa_store.structs": None})
    @patch.dict(sys.modules, {"sqalchemy": None})
    @patch("ax.service.ax_client.TDBSettings", None)
    def test_no_sqa(self) -> None:
        # Make sure we couldn't import sqa_store.structs (this could happen when
        # SQLAlchemy is not installed).
        with self.assertRaises(ModuleNotFoundError):
            # pyre-fixme[21]: Could not find module
            #  `ax_client.storage.sqa_store.structs`.
            # @manual
            import ax_client.storage.sqa_store.structs  # noqa F401
        # Make sure we can still import ax_client.
        __import__("ax.service.ax_client")
        AxClient()  # Make sure we still can instantiate client w/o db settings.
        # DBSettings should be defined in `ax_client` now, but incorrectly typed
        # `db_settings` argument should still make instantiation fail.
        with self.assertRaisesRegex(ValueError, "`db_settings` argument should "):
            # pyre-fixme[6]: For 1st param expected `Optional[DBSettings]` but got
            #  `str`.
            AxClient(db_settings="badly_typed_db_settings")

    def test_plotting_validation(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x3", "type": "fixed", "value": 2, "value_type": "int"}
            ]
        )
        with self.assertRaisesRegex(ValueError, ".* there are no trials"):
            ax_client.get_contour_plot()
        with self.assertRaisesRegex(ValueError, ".* there are no trials"):
            ax_client.get_feature_importances()
        ax_client.get_next_trial()
        with self.assertRaisesRegex(ValueError, ".* less than 2 parameters"):
            ax_client.get_contour_plot()
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ]
        )
        ax_client.get_next_trial()
        with self.assertRaisesRegex(ValueError, "If `param_x` is provided"):
            ax_client.get_contour_plot(param_x="y")
        with self.assertRaisesRegex(ValueError, "If `param_x` is provided"):
            ax_client.get_contour_plot(param_y="y")
        with self.assertRaisesRegex(ValueError, 'Parameter "x3"'):
            ax_client.get_contour_plot(param_x="x3", param_y="x3")
        with self.assertRaisesRegex(ValueError, 'Parameter "x4"'):
            ax_client.get_contour_plot(param_x="x", param_y="x4")
        with self.assertRaisesRegex(ValueError, 'Metric "nonexistent"'):
            ax_client.get_contour_plot(
                param_x="x", param_y="y", metric_name="nonexistent"
            )
        with self.assertRaisesRegex(UnsupportedPlotError, "Could not obtain contour"):
            ax_client.get_contour_plot(
                param_x="x", param_y="y", metric_name="objective"
            )
        with self.assertRaisesRegex(ValueError, "Could not obtain feature"):
            ax_client.get_feature_importances()

    def test_sqa_storage(self) -> None:
        init_test_engine_and_session_factory(force_init=True)
        config = SQAConfig()
        encoder = Encoder(config=config)
        decoder = Decoder(config=config)
        db_settings = DBSettings(encoder=encoder, decoder=decoder)
        ax_client = AxClient(db_settings=db_settings)
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            support_intermediate_data=True,
        )
        for _ in range(5):
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(
                trial_index=trial_index,
                # pyre-fixme[6]: For 2nd param expected `Union[List[Tuple[Dict[str, U...
                raw_data=[({"t": 0}, {"branin": (branin(*parameters.values()), 0.0)})],
            )
        gs = ax_client.generation_strategy
        ax_client = AxClient(db_settings=db_settings)
        ax_client.load_experiment_from_database("test_experiment")
        # Some fields of the reloaded GS are not expected to be set (both will be
        # set during next model fitting call), so we unset them on the original GS as
        # well.
        gs._unset_non_persistent_state_fields()
        ax_client.generation_strategy._unset_non_persistent_state_fields()
        self.assertEqual(gs, ax_client.generation_strategy)
        with self.assertRaises(ValueError):
            # Overwriting existing experiment.
            ax_client.create_experiment(
                name="test_experiment",
                parameters=[
                    {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                    {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
                ],
            )
        with self.assertRaises(ValueError):
            # Overwriting existing experiment with overwrite flag with present
            # DB settings. This should fail as we no longer allow overwriting
            # experiments stored in the DB.
            ax_client.create_experiment(
                name="test_experiment",
                parameters=[{"name": "x", "type": "range", "bounds": [-5.0, 10.0]}],
                overwrite_existing_experiment=True,
            )
        # Original experiment should still be in DB and not have been overwritten.
        self.assertEqual(len(ax_client.experiment.trials), 5)

        # Attach an early stopped trial.
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.update_running_trial_with_intermediate_data(
            trial_index=trial_index,
            # pyre-fixme[6]: For 2nd param expected `Union[List[Tuple[Dict[str, U...
            raw_data=[({"t": 0}, {"branin": (branin(*parameters.values()), 0.0)})],
        )
        ax_client.stop_trial_early(trial_index=trial_index)

        # Reload experiment and check that trial status is accurate.
        ax_client_new = AxClient(db_settings=db_settings)
        ax_client_new.load_experiment_from_database("test_experiment")
        self.assertEqual(
            ax_client.experiment.trials_by_status,
            ax_client_new.experiment.trials_by_status,
        )

    def test_overwrite(self) -> None:
        init_test_engine_and_session_factory(force_init=True)
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
        )

        # Log a trial
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(
            trial_index=trial_index,
            # pyre-fixme[6]: For 2nd param expected `Union[List[Tuple[Dict[str, Union...
            raw_data=branin(*parameters.values()),
        )

        with self.assertRaises(ValueError):
            # Overwriting existing experiment.
            ax_client.create_experiment(
                name="test_experiment",
                parameters=[
                    {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                    {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
                ],
            )
        # Overwriting existing experiment with overwrite flag.
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
            ],
            overwrite_existing_experiment=True,
        )
        # There should be no trials, as we just put in a fresh experiment.
        self.assertEqual(len(ax_client.experiment.trials), 0)

        # Log a trial
        parameters, trial_index = ax_client.get_next_trial()
        self.assertIn("x1", parameters.keys())
        self.assertIn("x2", parameters.keys())
        ax_client.complete_trial(
            trial_index=trial_index,
            # pyre-fixme[6]: For 2nd param expected `Union[List[Tuple[Dict[str, Union...
            raw_data=branin(*parameters.values()),
        )

    def test_fixed_random_seed_reproducibility(self) -> None:
        ax_client = AxClient(random_seed=RANDOM_SEED)
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ]
        )
        for _ in range(5):
            params, idx = ax_client.get_next_trial()
            # pyre-fixme[6]: For 2nd param expected `Union[List[Tuple[Dict[str, Union...
            ax_client.complete_trial(idx, branin(params.get("x"), params.get("y")))
        trial_parameters_1 = [
            # pyre-fixme[16]: `BaseTrial` has no attribute `arm`.
            t.arm.parameters
            for t in ax_client.experiment.trials.values()
        ]
        ax_client = AxClient(random_seed=RANDOM_SEED)
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ]
        )
        for _ in range(5):
            params, idx = ax_client.get_next_trial()
            # pyre-fixme[6]: For 2nd param expected `Union[List[Tuple[Dict[str, Union...
            ax_client.complete_trial(idx, branin(params.get("x"), params.get("y")))
        trial_parameters_2 = [
            t.arm.parameters for t in ax_client.experiment.trials.values()
        ]
        self.assertEqual(trial_parameters_1, trial_parameters_2)

    def test_init_position_saved(self) -> None:
        ax_client = AxClient(random_seed=RANDOM_SEED)
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            name="sobol_init_position_test",
        )
        for _ in range(4):
            # For each generated trial, snapshot the client before generating it,
            # then recreate client, regenerate the trial and compare the trial
            # generated before and after snapshotting. If the state of Sobol is
            # recorded correctly, the newly generated trial will be the same as
            # the one generated before the snapshotting.
            serialized = ax_client.to_json_snapshot()
            params, idx = ax_client.get_next_trial()
            ax_client = AxClient.from_json_snapshot(serialized)
            with self.subTest(ax=ax_client, params=params, idx=idx):
                new_params, new_idx = ax_client.get_next_trial()
                # Sobol "init_position" setting should be saved on the generator run.
                self.assertEqual(
                    # pyre-fixme[16]: `BaseTrial` has no attribute `_generator_run`.
                    ax_client.experiment.trials[
                        idx
                    ]._generator_run._model_state_after_gen["init_position"],
                    idx + 1,
                )
                self.assertEqual(params, new_params)
                self.assertEqual(idx, new_idx)
            # pyre-fixme[6]: For 2nd param expected `Union[List[Tuple[Dict[str, Union...
            ax_client.complete_trial(idx, branin(params.get("x"), params.get("y")))

    def test_unnamed_experiment_snapshot(self) -> None:
        ax_client = AxClient(random_seed=RANDOM_SEED)
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ]
        )
        serialized = ax_client.to_json_snapshot()
        ax_client = AxClient.from_json_snapshot(serialized)
        self.assertIsNone(ax_client.experiment._name)

    @patch(
        "ax.adapter.random.RandomAdapter._predict",
        autospec=True,
        return_value=[get_observation1trans(first_metric_name="branin").data],
    )
    def test_get_model_predictions(self, _predict) -> None:
        ax_client = get_branin_optimization()
        ax_client.get_next_trial()
        ax_client.complete_trial(0, {"branin": (5.0, 0.5)})
        self.assertEqual(ax_client.get_model_predictions(), {0: {"branin": (9.0, 1.0)}})

    def test_get_model_predictions_no_next_trial_all_trials(self) -> None:
        ax_client = _set_up_client_for_get_model_predictions_no_next_trial()
        _attach_completed_trials(ax_client)
        _attach_not_completed_trials(ax_client)

        all_predictions_dict = ax_client.get_model_predictions()
        # Expect all 4 trial predictions (2 completed + 2 not completed)
        self.assertEqual(len(all_predictions_dict), 4)
        # Expect two metrics (i.e. not filtered) per trial
        self.assertEqual(len(all_predictions_dict[0].keys()), 2)

    def test_get_model_predictions_no_next_trial_no_completed_trial(self) -> None:
        ax_client = _set_up_client_for_get_model_predictions_no_next_trial()
        _attach_not_completed_trials(ax_client)

        with self.assertRaisesRegex(
            DataRequiredError, "At least one trial must be completed"
        ):
            ax_client.get_model_predictions()

    def test_get_model_predictions_no_next_trial_filtered(self) -> None:
        ax_client = _set_up_client_for_get_model_predictions_no_next_trial()
        _attach_completed_trials(ax_client)
        _attach_not_completed_trials(ax_client)

        all_predictions_dict = ax_client.get_model_predictions(
            metric_names=["test_metric1"]
        )
        # Expect only one metric (i.e. filteres from two metrics) per trial
        self.assertEqual(len(all_predictions_dict[0]), 1)

    def test_get_model_predictions_no_next_trial_in_sample(self) -> None:
        ax_client = _set_up_client_for_get_model_predictions_no_next_trial()
        _attach_completed_trials(ax_client)
        _attach_not_completed_trials(ax_client)

        in_sample_predictions_dict = ax_client.get_model_predictions(
            include_out_of_sample=False
        )
        # Expect only 2 (completed) trial predictions
        self.assertEqual(len(in_sample_predictions_dict), 2)

    def test_get_model_predictions_no_next_trial_parameterizations(self) -> None:
        ax_client = _set_up_client_for_get_model_predictions_no_next_trial()
        _attach_completed_trials(ax_client)

        parameterizations = {
            18: {"x1": 0.3, "x2": 0.5},
            19: {"x1": 0.4, "x2": 0.5},
            20: {"x1": 0.8, "x2": 0.5},
        }
        parameterization_predictions_dict = ax_client.get_model_predictions(
            # pyre-ignore [6]
            parameterizations=parameterizations
        )
        # Expect predictions for only 3 input parameterizations,
        # and no trial predictions
        self.assertEqual(len(parameterization_predictions_dict), 3)

    def test_get_model_predictions_for_parameterization_no_next_trial(self) -> None:
        ax_client = _set_up_client_for_get_model_predictions_no_next_trial()
        _attach_completed_trials(ax_client)

        parameterizations = [
            {"x1": 0.3, "x2": 0.5},
            {"x1": 0.4, "x2": 0.5},
            {"x1": 0.8, "x2": 0.5},
        ]
        predictions_list = ax_client.get_model_predictions_for_parameterizations(
            # pyre-ignore [6]
            parameterizations=parameterizations
        )
        self.assertEqual(len(predictions_list), 3)

    def test_deprecated_save_load_method_errors(self) -> None:
        ax_client = AxClient()
        with self.assertRaises(NotImplementedError):
            ax_client.save()
        with self.assertRaises(NotImplementedError):
            ax_client.load()
        with self.assertRaises(NotImplementedError):
            ax_client.load_experiment("test_experiment")
        with self.assertRaises(NotImplementedError):
            ax_client.get_recommended_max_parallelism()

    def test_find_last_trial_with_parameterization(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            objectives={"branin": ObjectiveProperties(minimize=True)},
        )
        params, trial_idx = ax_client.get_next_trial()
        found_trial_idx = ax_client._find_last_trial_with_parameterization(
            parameterization=params
        )
        self.assertEqual(found_trial_idx, trial_idx)
        # Check that it's indeed the _last_ trial with params that is found.
        _, new_trial_idx = ax_client.attach_trial(parameters=params)
        found_trial_idx = ax_client._find_last_trial_with_parameterization(
            parameterization=params
        )
        self.assertEqual(found_trial_idx, new_trial_idx)
        with self.assertRaisesRegex(ValueError, "No .* matches"):
            found_trial_idx = ax_client._find_last_trial_with_parameterization(
                parameterization={k: v + 1.0 for k, v in params.items()}
            )

    def test_verify_parameterization(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            objectives={"branin": ObjectiveProperties(minimize=True)},
        )
        params, trial_idx = ax_client.get_next_trial()
        self.assertTrue(
            ax_client.verify_trial_parameterization(
                trial_index=trial_idx, parameterization=params
            )
        )
        # Make sure it still works if ordering in the parameterization is diff.
        self.assertTrue(
            ax_client.verify_trial_parameterization(
                trial_index=trial_idx,
                parameterization={k: params[k] for k in reversed(list(params.keys()))},
            )
        )
        self.assertFalse(
            ax_client.verify_trial_parameterization(
                trial_index=trial_idx,
                parameterization={k: v + 1.0 for k, v in params.items()},
            )
        )

    @patch(
        "ax.core.experiment.Experiment.new_trial",
        side_effect=RuntimeError("cholesky_cpu error - bad matrix"),
    )
    def test_annotate_exception(self, _) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            objectives={"branin": ObjectiveProperties(minimize=True)},
        )
        with self.assertRaisesRegex(
            expected_exception=RuntimeError,
            expected_regex="Cholesky errors typically occur",
        ):
            ax_client.get_next_trial()

    @patch(
        f"{get_pareto_optimal_parameters.__module__}.predicted_pareto",
        wraps=predicted_pareto,
    )
    @patch(
        f"{get_pareto_optimal_parameters.__module__}.observed_pareto",
        wraps=observed_pareto,
    )
    @mock_botorch_optimize
    def helper_test_get_pareto_optimal_points(
        self,
        mock_observed_pareto: Mock,
        mock_predicted_pareto: Mock,
        outcome_constraints: list[str] | None = None,
    ) -> None:
        ax_client, branin_currin = get_branin_currin_optimization_with_N_sobol_trials(
            num_trials=20, outcome_constraints=outcome_constraints
        )
        ax_client.fit_model()
        self.assertEqual(
            ax_client.generation_strategy._curr.generator_spec_to_gen_from.model_key,
            "BoTorch",
        )

        # Check calling get_best_parameters fails (user must call
        # get_pareto_optimal_parameters).
        with self.assertRaisesRegex(
            NotImplementedError, "Please use `get_pareto_optimal_parameters`"
        ):
            ax_client.get_best_parameters()
        with self.assertRaisesRegex(
            NotImplementedError, "Please use `get_pareto_optimal_parameters`"
        ):
            ax_client.get_best_trial()

        # Via inspect of ax_client.experiment.lookup_data().df, we can see that
        # the only trial generating data within the thresholds is 14
        # So that's the only one that can be on the Pareto frontier.
        solution = 14

        # Check model-predicted Pareto frontier using model on GS.
        # NOTE: model predictions are very poor due to `mock_botorch_optimize`.
        # This overwrites the `predict` call to return the original observations,
        # while testing the rest of the code as if we're using predictions.
        # pyre-fixme[16]: `Optional` has no attribute `model`.
        model = ax_client.generation_strategy.adapter.generator
        ys = model.surrogate.training_data[0].Y
        with patch.object(
            model, "predict", return_value=(ys, torch.zeros(*ys.shape, ys.shape[-1]))
        ):
            predicted_pareto = ax_client.get_pareto_optimal_parameters()
        # Since we're just using actual values as predicted, the solution should
        # be the same as in the observed case.
        self.assertEqual(sorted(predicted_pareto.keys()), [solution])
        observed_pareto = ax_client.get_pareto_optimal_parameters(
            use_model_predictions=False
        )
        # Check that we did not specify objective threshold overrides (because we
        # did not have to infer them)
        self.assertIsNone(
            mock_predicted_pareto.call_args.kwargs.get("objective_thresholds")
        )
        # Observed Pareto values should be better than the reference point.
        for obs in observed_pareto.values():
            branin: float = obs[1][0]["branin"]
            currin: float = obs[1][0]["currin"]
            self.assertGreater(branin, branin_currin.ref_point[0].item())
            self.assertGreater(currin, branin_currin.ref_point[1].item())
            if outcome_constraints is not None:
                self.assertEqual(branin + currin, obs[1][0]["c"])

        self.assertEqual(sorted(observed_pareto.keys()), [solution])
        # Check that we did not specify objective threshold overrides (because we
        # did not have to infer them)
        self.assertIsNone(mock_observed_pareto.call_args[1].get("objective_thresholds"))

    def test_get_pareto_optimal_points(self) -> None:
        for outcome_constraints in [None, ["c <= 100.0"]]:
            with self.subTest(outcome_constraints=outcome_constraints):
                self.helper_test_get_pareto_optimal_points(
                    outcome_constraints=outcome_constraints
                )

    def helper_test_get_pareto_optimal_points_from_sobol_step(
        self, minimize: bool, outcome_constraints: list[str] | None = None
    ) -> None:
        ax_client, _ = get_branin_currin_optimization_with_N_sobol_trials(
            num_trials=20, minimize=minimize, outcome_constraints=outcome_constraints
        )
        self.assertEqual(
            ax_client.generation_strategy._curr.generator_spec_to_gen_from.model_key,
            "Sobol",
        )

        cfg = none_throws(ax_client.experiment.optimization_config)
        assert isinstance(cfg, MultiObjectiveOptimizationConfig)
        thresholds = np.array([t.bound for t in cfg.objective_thresholds])

        with with_rng_seed(seed=RANDOM_SEED):
            predicted_pareto = ax_client.get_pareto_optimal_parameters()

        # For the predicted frontier, we don't know the solution a priori, so let's
        # check it
        for _, (point, _) in predicted_pareto.values():
            values = np.array([point["branin"], point["currin"]])
            within_threshold = (
                (thresholds > values) if minimize else (thresholds < values)
            )
            self.assertTrue(within_threshold.all())
            # ensure that predictions are in the original, not transformed, space
            self.assertTrue((values > 0).all() if minimize else (values < 0).all())

        # Answer known by inspection, see
        # `test_get_pareto_optimal_points_objective_threshold_inference`,
        # which has the same data

        # predicted_pareto is in format
        # {trial_index -> (parameterization, (means, covariances))}

        frontier_means = [elt[1][0] for elt in predicted_pareto.values()]
        frontier_means_arr = np.array(
            [[elt["branin"], elt["currin"]] for elt in frontier_means]
        )

        for point in frontier_means_arr:
            improvement = (
                point - frontier_means_arr if minimize else frontier_means_arr - point
            )
            has_dominating_points = (improvement > 0).all(1).any()
            self.assertFalse(has_dominating_points)
            within_threshold = (
                (point < thresholds) if minimize else (point > thresholds)
            )
            self.assertTrue(within_threshold.all())

        observed_pareto = ax_client.get_pareto_optimal_parameters(
            use_model_predictions=False
        )
        self.assertEqual(len(observed_pareto), 1)
        idx_of_frontier_point = 14
        self.assertEqual(next(iter(observed_pareto.keys())), idx_of_frontier_point)

        # Check that the data in the frontier matches the observed data
        # (it should be in the original, un-transformed space)
        input_data = (
            ax_client.experiment.fetch_data([idx_of_frontier_point]).df["mean"].values
        )
        pareto_y = observed_pareto[idx_of_frontier_point][1][0]
        pareto_y_list = [pareto_y["branin"], pareto_y["currin"]]
        if "c" in pareto_y.keys():
            pareto_y_list.append(pareto_y["c"])
        self.assertTrue((input_data == pareto_y_list).all())

    # Part 1/3 of tests run by helper_test_get_pareto_optimal_points_from_sobol_step
    @mock_botorch_optimize
    def test_get_pareto_optimal_points_from_sobol_step_no_constraint(self) -> None:
        outcome_constraints = None
        for minimize in [False, True]:
            self.helper_test_get_pareto_optimal_points_from_sobol_step(
                minimize=minimize, outcome_constraints=outcome_constraints
            )

    # Part 2/3 of tests run by helper_test_get_pareto_optimal_points_from_sobol_step
    @mock_botorch_optimize
    def test_get_pareto_optimal_points_from_sobol_step_with_constraint_minimize_true(
        self,
    ) -> None:
        self.helper_test_get_pareto_optimal_points_from_sobol_step(
            minimize=True, outcome_constraints=["c <= 100.0"]
        )

    # Part 3/3 of tests run by helper_test_get_pareto_optimal_points_from_sobol_step
    @mock_botorch_optimize
    def test_get_pareto_optimal_points_from_sobol_step_with_constraint_minimize_false(
        self,
    ) -> None:
        self.helper_test_get_pareto_optimal_points_from_sobol_step(
            minimize=False, outcome_constraints=["c <= 100.0"]
        )

    @patch(
        f"{get_pareto_optimal_parameters.__module__}.predicted_pareto",
        wraps=predicted_pareto,
    )
    @patch(
        f"{get_pareto_optimal_parameters.__module__}.observed_pareto",
        wraps=observed_pareto,
    )
    @mock_botorch_optimize
    def test_get_pareto_optimal_points_objective_threshold_inference(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        mock_observed_pareto,
        # pyre-fixme[2]: Parameter must be annotated.
        mock_predicted_pareto,
    ) -> None:
        ax_client, _ = get_branin_currin_optimization_with_N_sobol_trials(
            num_trials=20, include_objective_thresholds=False
        )
        ax_client.generation_strategy._maybe_transition_to_next_node()
        ax_client.generation_strategy._curr._fit(experiment=ax_client.experiment)
        with with_rng_seed(seed=RANDOM_SEED):
            predicted_pareto = ax_client.get_pareto_optimal_parameters()

        # Check that we specified objective threshold overrides (because we
        # inferred them)
        self.assertIsNotNone(
            mock_predicted_pareto.call_args[1].get("objective_thresholds")
        )
        mock_predicted_pareto.reset_mock()
        mock_observed_pareto.assert_not_called()
        self.assertGreater(len(predicted_pareto), 0)

        with with_rng_seed(seed=RANDOM_SEED):
            observed_pareto = ax_client.get_pareto_optimal_parameters(
                use_model_predictions=False
            )
        # Check that we specified objective threshold overrides (because we
        # inferred them)
        self.assertIsNotNone(
            mock_observed_pareto.call_args[1].get("objective_thresholds")
        )
        mock_predicted_pareto.assert_not_called()
        self.assertGreater(len(observed_pareto), 0)

    # Part 1/2 of tests run by helper_test_get_pareto_optimal_parameters_simple
    @mock_botorch_optimize
    def test_get_pareto_optimal_parameters_simple_with_minimize_false(self) -> None:
        minimize = False
        for use_y0_threshold, use_y2_constraint in product(
            [False, True], [False, True]
        ):
            self.helper_test_get_pareto_optimal_parameters_simple(
                minimize=minimize,
                use_y0_threshold=use_y0_threshold,
                use_y2_constraint=use_y2_constraint,
            )

    # Part 2/2 of tests run by helper_test_get_pareto_optimal_parameters_simple
    @mock_botorch_optimize
    def test_get_pareto_optimal_parameters_simple_with_minimize_true(self) -> None:
        minimize = True
        for use_y0_threshold, use_y2_constraint in product(
            [False, True], [False, True]
        ):
            self.helper_test_get_pareto_optimal_parameters_simple(
                minimize=minimize,
                use_y0_threshold=use_y0_threshold,
                use_y2_constraint=use_y2_constraint,
            )

    def helper_test_get_pareto_optimal_parameters_simple(
        self, minimize: bool, use_y0_threshold: bool, use_y2_constraint: bool
    ) -> None:
        """
        Construct a simple Pareto problem with just three known points.

        y-points (y0, y1, y2) are [(10, 12, 11), (11, 10, 11), (12, 11, 10)], with
        x-points [0, 1, 2], respectively. y1 and y2 are objectives and y3 is an
        additional metric. Points are not centered on zero to ensure that transforms
        will not affect the results.

        Solutions:
        - With no objectives or constraints, frontier is {0, 2} (since 2 dominates 1).
        - 2 dominates 1, unless 2 violates a threshold or constraint
        - use_y0_threshold eliminates 0
        - use_y2_constraint eliminates 2
        - 'minimize' shouldn't affect the solution since it reverses everything
        """

        def _get_parameterizations_from_pareto_frontier(
            pareto: dict[int, tuple[TParameterization, TModelPredictArm]],
        ) -> set[TParamValue]:
            return {tup[0]["x"] for tup in pareto.values()}

        # minimize doesn't affect solution
        def get_solution(use_y0_threshold: bool, use_y2_constraint: bool) -> set[int]:
            if use_y0_threshold:
                if use_y2_constraint:
                    return {1}
                return {2}
            if use_y2_constraint:
                return {0, 1}
            return {0, 2}

        ax_client = get_client_with_simple_discrete_moo_problem(
            minimize=minimize,
            use_y0_threshold=use_y0_threshold,
            use_y2_constraint=use_y2_constraint,
        )

        pareto_obs = ax_client.get_pareto_optimal_parameters(
            use_model_predictions=False
        )
        sol = get_solution(
            use_y0_threshold=use_y0_threshold,
            use_y2_constraint=use_y2_constraint,
        )

        self.assertSetEqual(
            sol, _get_parameterizations_from_pareto_frontier(pareto_obs)
        )
        pareto_mod = ax_client.get_pareto_optimal_parameters(use_model_predictions=True)
        # since this is a noise-free problem, using predicted values shouldn't
        # change the answer
        self.assertEqual(len(sol), len(pareto_mod))
        self.assertSetEqual(
            sol, _get_parameterizations_from_pareto_frontier(pareto_mod)
        )

        # take another step. This will change the generation strategy from
        # Sobol to MOO. Shouldn't affect results since we already had data
        # on all 3 points.
        parameterization, trial_index = ax_client.get_next_trial()
        x = parameterization["x"]

        metrics = y_values_for_simple_discrete_moo_problem[x]
        if minimize:
            metrics = [-m for m in metrics]
        y0, y1, y2 = metrics
        raw_data: TEvaluationOutcome = {
            "y0": (y0, 0.0),
            "y1": (y1, 0.0),
            "y2": (y2, 0.0),
        }

        ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)

        # Check frontier again
        pareto_obs = ax_client.get_pareto_optimal_parameters(use_model_predictions=True)
        self.assertSetEqual(
            sol, _get_parameterizations_from_pareto_frontier(pareto_obs)
        )
        pareto_mod = ax_client.get_pareto_optimal_parameters(use_model_predictions=True)
        self.assertSetEqual(
            sol, _get_parameterizations_from_pareto_frontier(pareto_mod)
        )

    @mock_botorch_optimize
    def test_get_hypervolume(self) -> None:
        # First check that hypervolume gets returned for observed data
        ax_client, _ = get_branin_currin_optimization_with_N_sobol_trials(num_trials=20)
        self.assertGreaterEqual(
            ax_client.get_hypervolume(use_model_predictions=False), 0
        )

        # Cannot get predicted hypervolume with sobol model
        with self.assertRaisesRegex(ValueError, "is not of type TorchAdapter"):
            ax_client.get_hypervolume(use_model_predictions=True)

        # Run one more trial and check predicted hypervolume gets returned
        _, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(
            trial_index,
            raw_data={"branin": 0, "currin": 0},
        )

        self.assertGreaterEqual(ax_client.get_hypervolume(), 0)

    @mock_botorch_optimize
    def test_with_hss(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {
                    "name": "model",
                    "type": "choice",
                    "values": ["Linear", "XGBoost"],
                    "dependents": {
                        "Linear": ["learning_rate", "l2_reg_weight"],
                        "XGBoost": ["num_boost_rounds"],
                    },
                },
                {
                    "name": "learning_rate",
                    "type": "range",
                    "bounds": [0.001, 0.1],
                },
                {
                    "name": "l2_reg_weight",
                    "type": "range",
                    "bounds": [0.00001, 0.001],
                },
                {
                    "name": "num_boost_rounds",
                    "type": "range",
                    "bounds": [0, 15],
                },
            ],
            objectives={"objective": ObjectiveProperties(minimize=True)},
            choose_generation_strategy_kwargs={"num_initialization_trials": 2},
        )
        hss = assert_is_instance(
            ax_client.experiment.search_space, HierarchicalSearchSpace
        )
        self.assertTrue(hss.root.is_hierarchical)

        ax_client.attach_trial({"model": "XGBoost", "num_boost_rounds": 2})
        ax_client.attach_trial(
            {"model": "Linear", "learning_rate": 0.003, "l2_reg_weight": 0.0003}
        )

        with self.assertRaises(RuntimeError):
            # Violates hierarchical structure of the search space.
            ax_client.attach_trial({"model": "Linear", "num_boost_rounds": 2})

        for _ in range(4):
            params, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(trial_index=trial_index, raw_data=trial_index)
        # Make sure we actually tried a Botorch iteration and all the transforms it
        # applies.
        self.assertEqual(
            ax_client.generation_strategy._generator_runs[-1]._model_key, "BoTorch"
        )
        self.assertEqual(len(ax_client.experiment.trials), 6)
        ax_client.attach_trial(
            {"model": "Linear", "learning_rate": 0.001, "l2_reg_weight": 0.0001}
        )
        with self.assertWarnsRegex(RuntimeWarning, "out-of-design"):
            ax_client.attach_trial(
                {"model": "Linear", "learning_rate": 1, "l2_reg_weight": 0.0001}
            )

    def test_should_stop_trials_early(self) -> None:
        expected: dict[int, str | None] = {
            1: "Stopped due to testing.",
            3: "Stopped due to testing.",
        }
        ax_client = AxClient(
            early_stopping_strategy=DummyEarlyStoppingStrategy(expected)
        )
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            support_intermediate_data=True,
        )
        # pyre-fixme[6]: For 1st param expected `Set[int]` but got `List[int]`.
        actual = ax_client.should_stop_trials_early(trial_indices=[1, 2, 3])
        self.assertEqual(actual, expected)

    def test_stop_trial_early(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            support_intermediate_data=True,
        )
        parameters, idx = ax_client.get_next_trial()
        ax_client.update_running_trial_with_intermediate_data(
            idx,
            # pyre-fixme[6]: For 2nd argument expected `Union[floating[typing...
            raw_data=[({"t": 0}, {"branin": (branin(*parameters.values()), 0.0)})],
        )
        ax_client.stop_trial_early(idx)
        trial = ax_client.get_trial(idx)
        self.assertTrue(trial.status.is_early_stopped)

    def test_estimate_early_stopping_savings(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            support_intermediate_data=True,
        )
        _, idx = ax_client.get_next_trial()
        ax_client.experiment.trials[idx].mark_early_stopped(unsafe=True)

        self.assertEqual(ax_client.estimate_early_stopping_savings(), 0)

    def test_max_parallelism_exception_when_early_stopping(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            support_intermediate_data=True,
        )

        exception = MaxParallelismReachedException(step_index=1, num_running=10)

        # pyre-fixme[53]: Captured variable `exception` is not annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def fake_new_trial(*args, **kwargs) -> None:
            raise exception

        ax_client.experiment.new_trial = fake_new_trial

        # Without early stopping.
        with self.assertRaises(MaxParallelismReachedException) as cm:
            ax_client.get_next_trial()
        # Assert Exception's message is unchanged.
        self.assertEqual(cm.exception.message, exception.message)

        # With early stopping.
        ax_client._early_stopping_strategy = DummyEarlyStoppingStrategy()
        # Assert Exception's message is augmented to mention early stopping.
        with self.assertRaisesRegex(MaxParallelismReachedException, ".*early.*stop"):
            ax_client.get_next_trial()

    def test_experiment_does_not_support_early_stopping(self) -> None:
        ax_client = AxClient(early_stopping_strategy=DummyEarlyStoppingStrategy())
        with self.assertRaisesRegex(ValueError, ".*`support_intermediate_data=True`.*"):
            ax_client.create_experiment(
                parameters=[
                    {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                    {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
                ],
                support_intermediate_data=False,
            )

    def test_torch_device(self) -> None:
        device = torch.device("cpu")
        with self.assertWarnsRegex(RuntimeWarning, "a `torch_device` were specified."):
            AxClient(
                generation_strategy=GenerationStrategy(
                    [GenerationStep(Generators.SOBOL, num_trials=3)]
                ),
                torch_device=device,
            )
        ax_client = get_branin_optimization(torch_device=device)
        gpei_step_kwargs = ax_client.generation_strategy._steps[1].model_kwargs
        self.assertEqual(gpei_step_kwargs["torch_device"], device)

    def test_repr_function(
        self,
    ) -> None:
        ax_client = AxClient()
        experiment_name = "test_experiment"
        ax_client.create_experiment(
            name=experiment_name,
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
        )
        self.assertEqual(
            ax_client.__repr__(), f"AxClient(experiment=Experiment({experiment_name}))"
        )

    def test_gen_fixed_features(self) -> None:
        ax_client = AxClient(random_seed=RANDOM_SEED)
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            name="fixed_features",
        )
        with mock.patch.object(
            GenerationStrategy,
            "gen_single_trial",
            wraps=ax_client.generation_strategy.gen_single_trial,
        ) as mock_gen:
            with self.subTest("fixed_features is None"):
                ax_client.get_next_trial()
                call_kwargs = mock_gen.call_args_list[0][1]
                ff = call_kwargs["fixed_features"]
                self.assertIsNone(ff)
            with self.subTest("fixed_features is set"):
                fixed_features = FixedFeatures(
                    parameters={"x": 0.0, "y": 5.0}, trial_index=0
                )
                ax_client.get_next_trial(fixed_features=fixed_features)
                call_kwargs = mock_gen.call_args_list[1][1]
                ff = call_kwargs["fixed_features"]
                self.assertEqual(ff.parameters, fixed_features.parameters)
                self.assertEqual(ff.trial_index, 0)

    def test_SingleTaskGP_log_unordered_categorical_parameters(self) -> None:
        logs = []

        ax_client = AxClient(random_seed=0)
        params = [
            {
                "name": f"x{i + 1}",
                "type": "range",
                "bounds": [*Branin._domain[i]],
                "value_type": "float",
                "log_scale": False,
            }
            for i in range(2)
        ]

        with mock.patch(
            "ax.generation_strategy.dispatch_utils.logger.info",
            side_effect=(lambda log: logs.append(log)),
        ):
            ax_client.create_experiment(
                name="branin_test_experiment",
                # pyre-fixme[6]: for argument `parameters`, expected
                # `List[Dict[str, Union[None, Dict[str, List[str]],
                # Sequence[Union[None, bool, float, int, str]],
                # bool, float, int, str]]]`
                # but got `List[Dict[str, Union[List[int], bool, str]]]`
                parameters=params,
                objectives={"branin": ObjectiveProperties(minimize=True)},
            )
        found_no_log = False
        for log in logs:
            # This message is confusing because there
            # are no unordered categorical parameters.
            self.assertNotIn(
                "categories for the unordered categorical parameters.", log
            )

            if "are no unordered categorical parameters." in log:
                found_no_log = True

        self.assertTrue(found_no_log)

    def test_with_node_based_gs(self) -> None:
        sobol_gs = GenerationStrategy(
            name="Sobol",
            nodes=[
                GenerationNode(
                    node_name="Sobol",
                    generator_specs=[GeneratorSpec(generator_enum=Generators.SOBOL)],
                )
            ],
        )
        ax_client = get_branin_optimization(generation_strategy=sobol_gs)
        params, idx = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=idx, raw_data={"branin": (0, 0.0)})

        self.assertEqual(ax_client.generation_strategy.name, "Sobol")
        self.assertEqual(
            assert_is_instance(
                ax_client.experiment.trials[0], Trial
            )._generator_run._model_key,
            "Sobol",
        )
        with mock.patch(
            "ax.service.ax_client.optimization_trace_single_method"
        ) as mock_plot:
            ax_client.get_optimization_trace()
        mock_plot.assert_called_once()


# Utility functions for testing get_model_predictions without calling
# get_next_trial. Create Ax Client with an experiment where
# num_initial_trials kwarg is zero. Note that this kwarg is
# needed to be able to instantiate the model for the first time
# without calling get_next_trial().
def _set_up_client_for_get_model_predictions_no_next_trial() -> AxClient:
    ax_client = AxClient()
    ax_client.create_experiment(
        name="test_experiment",
        choose_generation_strategy_kwargs={"num_initialization_trials": 0},
        parameters=[
            {
                "name": "x1",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x2",
                "type": "range",
                "bounds": [0.1, 1.0],
            },
        ],
        objectives={"test_metric1": ObjectiveProperties(minimize=False)},
        outcome_constraints=["test_metric2 <= 1.5"],
    )

    return ax_client


# pyre-fixme[2]: Parameter must be annotated.
def _attach_completed_trials(ax_client) -> None:
    # Attach completed trials
    trial1 = {"x1": 0.1, "x2": 0.1}
    parameters, trial_index = ax_client.attach_trial(trial1)
    ax_client.complete_trial(
        trial_index=trial_index, raw_data=_evaluate_test_metrics(parameters)
    )

    trial2 = {"x1": 0.2, "x2": 0.1}
    parameters, trial_index = ax_client.attach_trial(trial2)
    ax_client.complete_trial(
        trial_index=trial_index, raw_data=_evaluate_test_metrics(parameters)
    )


# pyre-fixme[2]: Parameter must be annotated.
def _attach_not_completed_trials(ax_client) -> None:
    # Attach not yet completed trials
    trial3 = {"x1": 0.3, "x2": 0.1}
    parameters, trial_index = ax_client.attach_trial(trial3)

    trial4 = {"x1": 0.4, "x2": 0.1}
    parameters, trial_index = ax_client.attach_trial(trial4)


# Test metric evaluation method
# pyre-fixme[2]: Parameter must be annotated.
def _evaluate_test_metrics(parameters) -> dict[str, tuple[float, float]]:
    x = np.array([parameters.get(f"x{i + 1}") for i in range(2)])
    return {"test_metric1": (x[0] / x[1], 0.0), "test_metric2": (x[0] + x[1], 0.0)}
