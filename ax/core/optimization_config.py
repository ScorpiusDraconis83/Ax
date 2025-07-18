#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from itertools import groupby

from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.outcome_constraint import (
    ComparisonOp,
    ObjectiveThreshold,
    OutcomeConstraint,
    ScalarizedOutcomeConstraint,
)
from ax.core.risk_measures import RiskMeasure
from ax.exceptions.core import UserInputError
from ax.utils.common.base import Base
from pyre_extensions import assert_is_instance


TRefPoint = list[ObjectiveThreshold]

# Sentinels for default arguments when None is a valid input
_NO_OUTCOME_CONSTRAINTS = [
    OutcomeConstraint(Metric("", lower_is_better=True), ComparisonOp.GEQ, 0)
]
_NO_OBJECTIVE_THRESHOLDS = [ObjectiveThreshold(Metric("", lower_is_better=True), 0)]
_NO_RISK_MEASURE = RiskMeasure("", {})


class OptimizationConfig(Base):
    """An optimization configuration, which comprises an objective,
    outcome constraints and an optional risk measure.

    There is no minimum or maximum number of outcome constraints, but an
    individual metric can have at most two constraints--which is how we
    represent metrics with both upper and lower bounds.
    """

    def __init__(
        self,
        objective: Objective,
        outcome_constraints: list[OutcomeConstraint] | None = None,
        risk_measure: RiskMeasure | None = None,
    ) -> None:
        """Inits OptimizationConfig.

        Args:
            objective: Metric+direction to use for the optimization.
            outcome_constraints: Constraints on metrics.
            risk_measure: An optional risk measure, used for robust optimization.
                Must be used with a `RobustSearchSpace`.
        """
        constraints: list[OutcomeConstraint] = (
            [] if outcome_constraints is None else outcome_constraints
        )
        self._validate_transformed_optimization_config(
            objective=objective,
            outcome_constraints=constraints,
            risk_measure=risk_measure,
        )
        self._objective: Objective = objective
        self._outcome_constraints: list[OutcomeConstraint] = constraints
        self.risk_measure: RiskMeasure | None = risk_measure

    def clone(self) -> "OptimizationConfig":
        """Make a copy of this optimization config."""
        return self.clone_with_args()

    def clone_with_args(
        self,
        objective: Objective | None = None,
        outcome_constraints: None | (list[OutcomeConstraint]) = _NO_OUTCOME_CONSTRAINTS,
        risk_measure: RiskMeasure | None = _NO_RISK_MEASURE,
    ) -> "OptimizationConfig":
        """Make a copy of this optimization config."""
        objective = self.objective.clone() if objective is None else objective
        outcome_constraints = (
            [constraint.clone() for constraint in self.outcome_constraints]
            if outcome_constraints is _NO_OUTCOME_CONSTRAINTS
            else outcome_constraints
        )
        risk_measure = (
            self.risk_measure if risk_measure is _NO_RISK_MEASURE else risk_measure
        )

        return OptimizationConfig(
            objective=objective,
            outcome_constraints=outcome_constraints,
            risk_measure=risk_measure,
        )

    @property
    def objective(self) -> Objective:
        """Get objective."""
        return self._objective

    @objective.setter
    def objective(self, objective: Objective) -> None:
        """Set objective if not present in outcome constraints."""
        self._validate_transformed_optimization_config(
            objective, self.outcome_constraints, self.risk_measure
        )
        self._objective = objective

    @property
    def all_constraints(self) -> list[OutcomeConstraint]:
        """Get outcome constraints."""
        return self.outcome_constraints

    @property
    def outcome_constraints(self) -> list[OutcomeConstraint]:
        """Get outcome constraints."""
        return self._outcome_constraints

    @property
    def metrics(self) -> dict[str, Metric]:
        constraint_metrics = {
            oc.metric.name: oc.metric
            for oc in self.all_constraints
            if not isinstance(oc, ScalarizedOutcomeConstraint)
        }
        scalarized_constraint_metrics = {
            metric.name: metric
            for oc in self.all_constraints
            if isinstance(oc, ScalarizedOutcomeConstraint)
            for metric in oc.metrics
        }
        objective_metrics = {metric.name: metric for metric in self.objective.metrics}
        return {
            **constraint_metrics,
            **scalarized_constraint_metrics,
            **objective_metrics,
        }

    @property
    def is_moo_problem(self) -> bool:
        return self.objective is not None and isinstance(self.objective, MultiObjective)

    @outcome_constraints.setter
    def outcome_constraints(self, outcome_constraints: list[OutcomeConstraint]) -> None:
        """Set outcome constraints if valid, else raise."""
        self._validate_transformed_optimization_config(
            objective=self.objective,
            outcome_constraints=outcome_constraints,
            risk_measure=self.risk_measure,
        )
        self._outcome_constraints = outcome_constraints

    @staticmethod
    def _validate_transformed_optimization_config(
        objective: Objective,
        outcome_constraints: list[OutcomeConstraint] | None = None,
        risk_measure: RiskMeasure | None = None,
    ) -> None:
        """Ensure outcome constraints are valid and the risk measure is
        compatible with the objective.

        Either one or two outcome constraints can reference one metric.
        If there are two constraints, they must have different 'ops': one
            LEQ and one GEQ.
        If there are two constraints, the bound of the GEQ op must be less
            than the bound of the LEQ op.

        Args:
            objective: Metric+direction to use for the optimization.
            outcome_constraints: Constraints to validate.
            risk_measure: An optional risk measure to validate.
        """
        if type(objective) is MultiObjective:
            # Raise error on exact equality; `ScalarizedObjective` is OK
            raise ValueError(
                "OptimizationConfig does not support MultiObjective. "
                "Use MultiObjectiveOptimizationConfig instead."
            )
        outcome_constraints = outcome_constraints or []
        # Only vaidate `outcome_constraints`
        outcome_constraints = [
            constraint
            for constraint in outcome_constraints
            if isinstance(constraint, ScalarizedOutcomeConstraint) is False
        ]
        unconstrainable_metrics = objective.get_unconstrainable_metrics()
        OptimizationConfig._validate_outcome_constraints(
            unconstrainable_metrics=unconstrainable_metrics,
            outcome_constraints=outcome_constraints,
        )

    @staticmethod
    def _validate_outcome_constraints(
        unconstrainable_metrics: list[Metric],
        outcome_constraints: list[OutcomeConstraint],
    ) -> None:
        constraint_metrics = [
            constraint.metric.name for constraint in outcome_constraints
        ]
        for metric in unconstrainable_metrics:
            if metric.name in constraint_metrics:
                raise ValueError("Cannot constrain on objective metric.")

        def get_metric_name(oc: OutcomeConstraint) -> str:
            return oc.metric.name

        sorted_constraints = sorted(outcome_constraints, key=get_metric_name)
        for metric_name, constraints_itr in groupby(
            sorted_constraints, get_metric_name
        ):
            constraints: list[OutcomeConstraint] = list(constraints_itr)
            constraints_len = len(constraints)
            if constraints_len == 2:
                if constraints[0].op == constraints[1].op:
                    raise ValueError(f"Duplicate outcome constraints {metric_name}")
                lower_bound_idx = 0 if constraints[0].op == ComparisonOp.GEQ else 1
                upper_bound_idx = 1 - lower_bound_idx
                lower_bound = constraints[lower_bound_idx].bound
                upper_bound = constraints[upper_bound_idx].bound
                if lower_bound >= upper_bound:
                    raise ValueError(
                        f"Lower bound {lower_bound} is >= upper bound "
                        + f"{upper_bound} for {metric_name}"
                    )
            elif constraints_len > 2:
                raise ValueError(f"Duplicate outcome constraints {metric_name}")

    def __repr__(self) -> str:
        base_repr = (
            f"{self.__class__.__name__}("
            "objective=" + repr(self.objective) + ", "
            "outcome_constraints=" + repr(self.outcome_constraints)
        )
        if self.risk_measure is None:
            end_repr = ")"
        else:
            end_repr = ", risk_measure=" + repr(self.risk_measure) + ")"
        return base_repr + end_repr

    def __hash__(self) -> int:
        """Make the class hashable to support grouping of GeneratorRuns."""
        return hash(repr(self))


class MultiObjectiveOptimizationConfig(OptimizationConfig):
    """An optimization configuration for multi-objective optimization,
    which comprises multiple objective, outcome constraints, objective
    thresholds, and an optional risk measure.

    There is no minimum or maximum number of outcome constraints, but an
    individual metric can have at most two constraints--which is how we
    represent metrics with both upper and lower bounds.

    ObjectiveThresholds should be present for every objective. A good
    rule of thumb is to set them 10% below the minimum acceptable value
    for each metric.
    """

    def __init__(
        self,
        objective: MultiObjective | ScalarizedObjective,
        outcome_constraints: list[OutcomeConstraint] | None = None,
        objective_thresholds: list[ObjectiveThreshold] | None = None,
        risk_measure: RiskMeasure | None = None,
    ) -> None:
        """Inits OptimizationConfig.

        Args:
            objective: Metric+direction to use for the optimization. Should be either a
                MultiObjective or a ScalarizedObjective.
            outcome_constraints: Constraints on metrics.
            objective_thesholds: Thresholds objectives must exceed. Used for
                multi-objective optimization and for calculating frontiers
                and hypervolumes.
            risk_measure: An optional risk measure, used for robust optimization.
                Must be used with a `RobustSearchSpace`.
        """
        constraints: list[OutcomeConstraint] = (
            [] if outcome_constraints is None else outcome_constraints
        )
        objective_thresholds = objective_thresholds or []
        self._validate_transformed_optimization_config(
            objective=objective,
            outcome_constraints=constraints,
            objective_thresholds=objective_thresholds,
            risk_measure=risk_measure,
        )
        self._objective: MultiObjective | ScalarizedObjective = objective
        self._outcome_constraints: list[OutcomeConstraint] = constraints
        self._objective_thresholds: list[ObjectiveThreshold] = objective_thresholds
        self.risk_measure: RiskMeasure | None = risk_measure

    # pyre-fixme[14]: Inconsistent override.
    def clone_with_args(
        self,
        objective: MultiObjective | ScalarizedObjective | None = None,
        outcome_constraints: None | (list[OutcomeConstraint]) = _NO_OUTCOME_CONSTRAINTS,
        objective_thresholds: None
        | (list[ObjectiveThreshold]) = _NO_OBJECTIVE_THRESHOLDS,
        risk_measure: RiskMeasure | None = _NO_RISK_MEASURE,
    ) -> "MultiObjectiveOptimizationConfig":
        """Make a copy of this optimization config."""
        objective = self.objective.clone() if objective is None else objective
        outcome_constraints = (
            [constraint.clone() for constraint in self.outcome_constraints]
            if outcome_constraints is _NO_OUTCOME_CONSTRAINTS
            else outcome_constraints
        )
        objective_thresholds = (
            [ot.clone() for ot in self.objective_thresholds]
            if objective_thresholds is _NO_OBJECTIVE_THRESHOLDS
            else objective_thresholds
        )
        risk_measure = (
            self.risk_measure if risk_measure is _NO_RISK_MEASURE else risk_measure
        )

        return MultiObjectiveOptimizationConfig(
            objective=objective,
            outcome_constraints=outcome_constraints,
            objective_thresholds=objective_thresholds,
            risk_measure=risk_measure,
        )

    @property
    def objective(self) -> MultiObjective | ScalarizedObjective:
        """Get objective."""
        return self._objective

    @objective.setter
    def objective(self, objective: MultiObjective | ScalarizedObjective) -> None:
        """Set objective if not present in outcome constraints."""
        self._validate_transformed_optimization_config(
            objective=objective,
            outcome_constraints=self.outcome_constraints,
            objective_thresholds=self.objective_thresholds,
            risk_measure=self.risk_measure,
        )
        self._objective = objective

    @property
    def all_constraints(self) -> list[OutcomeConstraint]:
        """Get all constraints and thresholds."""
        return self.outcome_constraints + self.objective_thresholds

    @property
    def objective_thresholds(self) -> list[ObjectiveThreshold]:
        """Get objective thresholds."""
        return self._objective_thresholds

    @objective_thresholds.setter
    def objective_thresholds(
        self, objective_thresholds: list[ObjectiveThreshold]
    ) -> None:
        """Set outcome constraints if valid, else raise."""
        self._validate_transformed_optimization_config(
            objective=self.objective,
            objective_thresholds=objective_thresholds,
            risk_measure=self.risk_measure,
        )
        self._objective_thresholds = objective_thresholds

    @property
    def objective_thresholds_dict(self) -> dict[str, ObjectiveThreshold]:
        """Get a mapping from objective metric name to the corresponding
        threshold.
        """
        return {ot.metric.name: ot for ot in self._objective_thresholds}

    @staticmethod
    def _validate_transformed_optimization_config(
        objective: Objective,
        outcome_constraints: list[OutcomeConstraint] | None = None,
        objective_thresholds: list[ObjectiveThreshold] | None = None,
        risk_measure: RiskMeasure | None = None,
    ) -> None:
        """Ensure outcome constraints are valid and the risk measure is
        compatible with the objective.

        Either one or two outcome constraints can reference one metric.
        If there are two constraints, they must have different 'ops': one
            LEQ and one GEQ.
        If there are two constraints, the bound of the GEQ op must be less
            than the bound of the LEQ op.

        Args:
            objective: Metric+direction to use for the optimization.
            outcome_constraints: Constraints to validate.
            objective_thresholds: Thresholds objectives must exceed.
            risk_measure: An optional risk measure to validate.
        """
        if not isinstance(objective, (MultiObjective, ScalarizedObjective)):
            raise TypeError(
                "`MultiObjectiveOptimizationConfig` requires an objective "
                "of type `MultiObjective` or `ScalarizedObjective`. "
                "Use `OptimizationConfig` instead if using a "
                "single-metric objective."
            )
        outcome_constraints = outcome_constraints or []
        objective_thresholds = objective_thresholds or []
        if isinstance(objective, MultiObjective):
            objectives_by_name = {obj.metric.name: obj for obj in objective.objectives}
            check_objective_thresholds_match_objectives(
                objectives_by_name=objectives_by_name,
                objective_thresholds=objective_thresholds,
            )

        unconstrainable_metrics = objective.get_unconstrainable_metrics()
        OptimizationConfig._validate_outcome_constraints(
            unconstrainable_metrics=unconstrainable_metrics,
            outcome_constraints=outcome_constraints,
        )

    def __repr__(self) -> str:
        base_repr = (
            f"{self.__class__.__name__}("
            "objective=" + repr(self.objective) + ", "
            "outcome_constraints=" + repr(self.outcome_constraints) + ", "
            "objective_thresholds=" + repr(self.objective_thresholds)
        )
        if self.risk_measure is None:
            end_repr = ")"
        else:
            end_repr = ", risk_measure=" + repr(self.risk_measure) + ")"
        return base_repr + end_repr


def check_objective_thresholds_match_objectives(
    objectives_by_name: dict[str, Objective],
    objective_thresholds: list[ObjectiveThreshold],
) -> None:
    """Error if thresholds on objective_metrics bound from the wrong direction or
    if there is a mismatch between objective thresholds and objectives.
    """
    obj_thresh_metrics = set()
    for threshold in objective_thresholds:
        metric_name = threshold.metric.name
        if metric_name not in objectives_by_name:
            raise UserInputError(
                f"Objective threshold {threshold} is on metric '{metric_name}', "
                f"but that metric is not among the objectives."
            )
        if metric_name in obj_thresh_metrics:
            raise UserInputError(
                "More than one objective threshold specified for metric "
                f"{metric_name}."
            )
        obj_thresh_metrics.add(metric_name)

        minimize = objectives_by_name[metric_name].minimize
        bounded_above = threshold.op == ComparisonOp.LEQ
        is_aligned = minimize == bounded_above
        if not is_aligned:
            raise UserInputError(
                f"Objective threshold on {metric_name} bounds from "
                f"{'above' if bounded_above else 'below'} "
                f"but {metric_name} is being "
                f"{'minimized' if minimize else 'maximized'}."
            )


class PreferenceOptimizationConfig(MultiObjectiveOptimizationConfig):
    def __init__(
        self,
        objective: MultiObjective,
        preference_profile_name: str,
        outcome_constraints: list[OutcomeConstraint] | None = None,
    ) -> None:
        """Inits PreferenceOptimizationConfig.

        Args:
            objective: Metric+direction to use for the optimization. Should be a
                MultiObjective.
            preference_profile_name: The name of the auxiliary experiment to use as the
                preference profile for the experiment. An auxiliary experiment with
                this name and purpose PE_EXPERIMENT should be attached to
                the experiment.
            outcome_constraints: Constraints on metrics. Not yet supported.
        """
        if outcome_constraints:
            raise NotImplementedError(
                "Outcome constraints are not yet supported in "
                "PreferenceOptimizationConfig"
            )

        # Call parent's __init__ with objective_thresholds=None and risk_measure=None
        super().__init__(
            objective=objective,
            outcome_constraints=outcome_constraints,
            objective_thresholds=None,
            risk_measure=None,
        )
        self.preference_profile_name = preference_profile_name

    # pyre-ignore[14]: Inconsistent override.
    def clone_with_args(
        self,
        objective: MultiObjective | None = None,
        preference_profile_name: str | None = None,
        outcome_constraints: list[OutcomeConstraint] | None = None,
    ) -> PreferenceOptimizationConfig:
        """Make a copy of this optimization config."""
        objective = (
            assert_is_instance(self.objective.clone(), MultiObjective)
            if objective is None
            else objective
        )

        preference_profile_name = (
            self.preference_profile_name
            if preference_profile_name is None
            else preference_profile_name
        )
        outcome_constraints = (
            [constraint.clone() for constraint in self.outcome_constraints]
            if outcome_constraints is None
            else outcome_constraints
        )

        return PreferenceOptimizationConfig(
            objective=objective,
            preference_profile_name=preference_profile_name,
            outcome_constraints=outcome_constraints,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            "objective=" + repr(self.objective) + ", "
            "preference_profile_name=" + repr(self.preference_profile_name) + ", "
            "outcome_constraints=" + repr(self.outcome_constraints)
        )
