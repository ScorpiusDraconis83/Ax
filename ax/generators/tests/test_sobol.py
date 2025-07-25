#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest import mock

import numpy as np
from ax.exceptions.core import SearchSpaceExhausted
from ax.generators.random.sobol import SobolGenerator
from ax.utils.common.testutils import TestCase
from botorch.utils.sampling import sample_polytope
from numpy import typing as npt


class SobolGeneratorTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tunable_param_bounds = (0.0, 1.0)
        self.fixed_param_bounds = (1.0, 100.0)

    def _create_bounds(self, n_tunable: int, n_fixed: int) -> list[tuple[float, float]]:
        tunable_bounds = [self.tunable_param_bounds] * n_tunable
        fixed_bounds = [self.fixed_param_bounds] * n_fixed
        return tunable_bounds + fixed_bounds

    def test_SobolGeneratorAllTunable(self) -> None:
        generator = SobolGenerator(seed=0)
        bounds = self._create_bounds(n_tunable=3, n_fixed=0)
        generated_points, weights = generator.gen(
            n=3, bounds=bounds, rounding_func=lambda x: x
        )
        self.assertEqual(np.shape(generated_points), (3, 3))
        np_bounds = np.array(bounds)
        self.assertTrue(np.all(generated_points >= np_bounds[:, 0]))
        self.assertTrue(np.all(generated_points <= np_bounds[:, 1]))
        self.assertTrue(np.all(weights == 1.0))
        state = generator._get_state()
        self.assertEqual(state.get("init_position"), 3)
        self.assertEqual(state.get("seed"), generator.seed)
        generator.gen(
            n=3,
            bounds=bounds,
            rounding_func=lambda x: x,
            generated_points=generated_points,
        )

    def test_SobolGeneratorFixedSpace(self) -> None:
        generator = SobolGenerator(seed=0, deduplicate=False)
        bounds = self._create_bounds(n_tunable=0, n_fixed=2)
        generated_points, _ = generator.gen(
            n=3,
            bounds=bounds,
            fixed_features={0: 1, 1: 2},
            rounding_func=lambda x: x,
        )
        self.assertEqual(np.shape(generated_points), (3, 2))
        np_bounds = np.array(bounds)
        self.assertTrue(np.all(generated_points >= np_bounds[:, 0]))
        self.assertTrue(np.all(generated_points <= np_bounds[:, 1]))
        # Should error out if deduplicating since there's only one feasible point.
        generator = SobolGenerator(seed=0, deduplicate=True)
        with self.assertRaisesRegex(SearchSpaceExhausted, "Rejection sampling"):
            generated_points, _ = generator.gen(
                n=3,
                bounds=bounds,
                fixed_features={0: 1, 1: 2},
                rounding_func=lambda x: x,
            )
        # But we can generate 1 point.
        generated_points, _ = generator.gen(
            n=1,
            bounds=bounds,
            fixed_features={0: 1, 1: 2},
            rounding_func=lambda x: x,
        )
        self.assertEqual(np.shape(generated_points), (1, 2))
        # Errors out if we have already generated the only point.
        with self.assertRaisesRegex(SearchSpaceExhausted, "Rejection sampling"):
            generator.gen(
                n=1,
                bounds=bounds,
                fixed_features={0: 1, 1: 2},
                rounding_func=lambda x: x,
                generated_points=generated_points,
            )

    def test_SobolGeneratorNoScramble(self) -> None:
        generator = SobolGenerator(scramble=False)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)
        generated_points, _ = generator.gen(
            n=3,
            bounds=bounds,
            fixed_features={fixed_param_index: 1},
            rounding_func=lambda x: x,
        )
        self.assertEqual(np.shape(generated_points), (3, 4))
        np_bounds = np.array(bounds)
        self.assertTrue(np.all(generated_points >= np_bounds[:, 0]))
        self.assertTrue(np.all(generated_points <= np_bounds[:, 1]))

    def test_SobolGeneratorOnline(self) -> None:
        # Verify that the generator will return the expected arms if called
        # one at a time.
        bulk_generator = SobolGenerator(seed=0)
        generator = SobolGenerator(seed=0)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)
        bulk_generated_points, _ = bulk_generator.gen(
            n=3,
            bounds=bounds,
            fixed_features={fixed_param_index: 1},
            rounding_func=lambda x: x,
        )
        np_bounds = np.array(bounds)
        all_generated = np.zeros((0, len(bounds)))
        for expected_points in bulk_generated_points:
            generated_points, weights = generator.gen(
                n=1,
                bounds=bounds,
                fixed_features={fixed_param_index: 1},
                rounding_func=lambda x: x,
                generated_points=all_generated,
            )
            all_generated = np.vstack((all_generated, generated_points))
            self.assertEqual(weights, [1])
            self.assertTrue(np.all(generated_points >= np_bounds[:, 0]))
            self.assertTrue(np.all(generated_points <= np_bounds[:, 1]))
            self.assertTrue(generated_points[..., -1] == 1)
            self.assertTrue(np.array_equal(expected_points, generated_points.flatten()))

    def test_SobolGeneratorWithOrderConstraints(self) -> None:
        # Enforce dim_0 <= dim_1 <= dim_2 <= dim_3.
        # Enforce both fixed and tunable constraints.
        generator = SobolGenerator(seed=0)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)
        A = np.array([[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]])
        b = np.array([0, 0, 0])
        generated_points, _ = generator.gen(
            n=3,
            bounds=bounds,
            linear_constraints=(A, b),
            fixed_features={fixed_param_index: 0.5},
            rounding_func=lambda x: x,
        )
        self.assertEqual(np.shape(generated_points), (3, 4))
        self.assertTrue(np.all(generated_points[..., -1] == 0.5))
        self.assertTrue(
            np.array_equal(
                np.sort(generated_points[..., :-1], axis=-1),
                generated_points[..., :-1],
            )
        )

    def test_SobolGeneratorWithLinearConstraints(self) -> None:
        # Enforce dim_0 <= dim_1 <= dim_2 <= dim_3.
        # Enforce both fixed and tunable constraints.
        generator = SobolGenerator(seed=0)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)
        A = np.array([[1, 1, 0, 0], [0, 1, 1, 0]])
        b = np.array([1, 1])

        generated_points, _ = generator.gen(
            n=3,
            bounds=bounds,
            linear_constraints=(A, b),
            fixed_features={fixed_param_index: 1},
            rounding_func=lambda x: x,
        )
        self.assertTrue(np.shape(generated_points) == (3, 4))
        self.assertTrue(np.all(generated_points[..., -1] == 1))
        self.assertTrue(np.all(generated_points @ A.transpose() <= b))

    def test_SobolGeneratorFallbackToPolytopeSampler(self) -> None:
        # Ten parameters with sum less than 1. In this example, the rejection
        # sampler gives a search space exhausted error.  Testing fallback to
        # polytope sampler when encountering this error.
        generator = SobolGenerator(seed=0, fallback_to_sample_polytope=True)
        bounds = self._create_bounds(n_tunable=10, n_fixed=0)
        A = np.ones((1, 10))
        b = np.array([1]).reshape((1, 1))

        def rounding_function(x: npt.NDArray) -> npt.NDArray:
            return x

        rounding_func = mock.Mock(wraps=rounding_function)

        with mock.patch(
            "ax.generators.random.base.logger.warning"
        ) as mock_logger, mock.patch(
            "botorch.utils.sampling.sample_polytope",
            wraps=sample_polytope,
        ) as wrapped_sampler:
            generated_points, _ = generator.gen(
                n=3,
                bounds=bounds,
                linear_constraints=(A, b),
                rounding_func=rounding_func,
            )
        # First call uses the original seed since no candidates are generated.
        self.assertEqual(wrapped_sampler.call_args.kwargs["seed"], 0)
        self.assertTrue(
            "exceeded specified maximum draws" in mock_logger.call_args.args[0]
        )
        self.assertTrue(np.shape(generated_points) == (3, 10))
        self.assertTrue(np.all(generated_points @ A.transpose() <= b))
        rounding_func.assert_called()

        rounding_func.reset_mock()
        with mock.patch(
            "botorch.utils.sampling.sample_polytope",
            wraps=sample_polytope,
        ) as wrapped_sampler:
            generator.gen(
                n=3,
                bounds=bounds,
                linear_constraints=(A, b),
                rounding_func=rounding_func,
                generated_points=generated_points,
            )
        # Second call uses seed 3 since 3 candidates are already generated.
        self.assertEqual(wrapped_sampler.call_args.kwargs["seed"], 3)
        rounding_func.assert_called()

    def test_SobolGeneratorFallbackToPolytopeSamplerWithFixedParam(self) -> None:
        # Ten parameters with sum less than 1. In this example, the rejection
        # sampler gives a search space exhausted error.  Testing fallback to
        # polytope sampler when encountering this error.
        generator = SobolGenerator(seed=0, fallback_to_sample_polytope=True)
        bounds = self._create_bounds(n_tunable=10, n_fixed=1)
        A = np.insert(np.ones((1, 10)), 10, 0, axis=1)
        b = np.array([1]).reshape((1, 1))
        generated_points, _ = generator.gen(
            n=3,
            bounds=bounds,
            linear_constraints=(A, b),
            fixed_features={10: 1},
            rounding_func=lambda x: x,
        )
        self.assertTrue(np.shape(generated_points) == (3, 11))
        self.assertTrue(np.all(generated_points[..., -1] == 1))
        self.assertTrue(np.all(generated_points @ A.transpose() <= b))

    def test_SobolGeneratorOnlineRestart(self) -> None:
        # Ensure a single batch generation can also equivalently done by
        # a partial generation, re-initialization of a new SobolGenerator,
        # and a final generation.
        generator = SobolGenerator(seed=0)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)
        generated_points_single_batch, _ = generator.gen(
            n=3,
            bounds=bounds,
            fixed_features={fixed_param_index: 1},
            rounding_func=lambda x: x,
        )

        generator_first_batch = SobolGenerator(seed=0)
        generated_points_first_batch, _ = generator_first_batch.gen(
            n=1,
            bounds=bounds,
            fixed_features={fixed_param_index: 1},
            rounding_func=lambda x: x,
        )
        generator_second_batch = SobolGenerator(
            init_position=generator_first_batch.init_position, seed=0
        )
        generated_points_second_batch, _ = generator_second_batch.gen(
            n=2,
            bounds=bounds,
            fixed_features={fixed_param_index: 1},
            rounding_func=lambda x: x,
            generated_points=generated_points_first_batch,
        )

        generated_points_two_trials = np.vstack(
            (generated_points_first_batch, generated_points_second_batch)
        )
        self.assertTrue(
            np.shape(generated_points_single_batch)
            == np.shape(generated_points_two_trials)
        )
        self.assertTrue(
            np.allclose(generated_points_single_batch, generated_points_two_trials)
        )

    def test_SobolGeneratorBadBounds(self) -> None:
        generator = SobolGenerator()
        with self.assertRaisesRegex(ValueError, "This generator operates on"):
            generator.gen(
                n=1,
                bounds=[(-1, 1)],
                rounding_func=lambda x: x,
            )

    def test_SobolGeneratorMaxDraws(self) -> None:
        generator = SobolGenerator(seed=0)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)
        with self.assertRaises(SearchSpaceExhausted):
            generator.gen(
                n=3,
                bounds=bounds,
                linear_constraints=(
                    np.array([[1, 1, 0, 0], [0, 1, 1, 0]]),
                    np.array([1, 1]),
                ),
                fixed_features={fixed_param_index: 1},
                model_gen_options={"max_rs_draws": 0},
                rounding_func=lambda x: x,
            )

    def test_SobolGeneratorDedupe(self) -> None:
        generator = SobolGenerator(seed=0, deduplicate=True)
        n_tunable = fixed_param_index = 3
        bounds = self._create_bounds(n_tunable=n_tunable, n_fixed=1)
        generated_points, _ = generator.gen(
            n=2,
            bounds=bounds,
            linear_constraints=(
                np.array([[1, 1, 0, 0], [0, 1, 1, 0]]),
                np.array([1, 1]),
            ),
            fixed_features={fixed_param_index: 1},
            rounding_func=lambda x: x,
        )
        self.assertEqual(len(generated_points), 2)
        generated_points, _ = generator.gen(
            n=1,
            bounds=bounds,
            linear_constraints=(
                np.array([[1, 1, 0, 0], [0, 1, 1, 0]]),
                np.array([1, 1]),
            ),
            fixed_features={fixed_param_index: 1},
            rounding_func=lambda x: x,
        )
        self.assertEqual(len(generated_points), 1)
