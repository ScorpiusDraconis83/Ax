#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
from unittest.mock import patch

import pandas as pd
from ax.core.data import clone_without_metrics, custom_data_class, Data
from ax.utils.common.testutils import TestCase
from ax.utils.common.timeutils import current_timestamp_in_millis

REPR_1000: str = (
    "Data(df=\n|    |   trial_index |   arm_name | metric_name   |   mean |   sem "
    "| start_time          | end_time            |\n"
    "|---:|--------------:|-----------:|:--------------|-------:|------:"
    "|:--------------------|:--------------------|\n"
    "|  0 |             1 |        0_0 | a             |    2   |   0.2 "
    "| 2018-01-01 00:00:00 | 2018-01-02 00:00:00 |\n"
    "|  1 |             1 |        0_0 | b             |    1.8 |   0.3 "
    "| 2018-01-01 00:00:00 | 2018-01-02 00:00:00 |\n"
    "|  2 |             1 |        0_1 | a             |    4   |   0.6 "
    "| 2018-01-01 00:00:00 | 2018-01-02 00:00:00 |\n"
    "|  3 |             1 |        0_1 | b             |    3.7 |   0.5 "
    "| 2018-01-01 00:00:00 | 2018-01-02 00:00:00 |\n"
    "|  4 |             1 |        0_2 | a             |    0.5 | nan   "
    "| 2018-01-01 00:00:00 | 2018-01-02 00:00:00 |\n"
    "|  5 |             1 |        0_2 | b             |    3   | nan   "
    "| 2018-01-01 00:00:00 | 2018-01-02 00:00:00 |)"
)

REPR_500: str = (
    "Data(df=\n|    |   trial_index |   arm_name | metric_name   |   mean |   sem "
    "| start_time          | end_time            |\n"
    "|---:|--------------:|-----------:|:--------------|-------:|------:"
    "|:--------------------|:--------------------|\n"
    "|  0 |             1 |        0_0 | a             |    2   |   0.2 "
    "| 2018-01-01 00:00:00 | 2018-01-02 00:00:00 |\n"
    "|  1 |             1 |        0_0 | b             |    1.8 |   0.3 "
    "| 2018-01-01 00:00:00 | 2018-01-02 00:00:00 |\n"
    "|  2 |             1 |        0_1 | a           ...)"
)


class DataTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.df_hash = "be6ca1edb2d83e08c460665476d32caa"
        self.df = pd.DataFrame(
            [
                {
                    "arm_name": "0_0",
                    "mean": 2.0,
                    "sem": 0.2,
                    "trial_index": 1,
                    "metric_name": "a",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_0",
                    "mean": 1.8,
                    "sem": 0.3,
                    "trial_index": 1,
                    "metric_name": "b",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_1",
                    "mean": 4.0,
                    "sem": 0.6,
                    "trial_index": 1,
                    "metric_name": "a",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_1",
                    "mean": 3.7,
                    "sem": 0.5,
                    "trial_index": 1,
                    "metric_name": "b",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_2",
                    "mean": 0.5,
                    "sem": None,
                    "trial_index": 1,
                    "metric_name": "a",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
                {
                    "arm_name": "0_2",
                    "mean": 3.0,
                    "sem": None,
                    "trial_index": 1,
                    "metric_name": "b",
                    "start_time": "2018-01-01",
                    "end_time": "2018-01-02",
                },
            ]
        )

    def test_Data(self) -> None:
        self.assertEqual(Data(), Data())
        data = Data(df=self.df)
        self.assertEqual(data, data)

        df = data.df
        self.assertEqual(
            float(df[df["arm_name"] == "0_0"][df["metric_name"] == "a"]["mean"].item()),
            2.0,
        )
        self.assertEqual(
            float(df[df["arm_name"] == "0_1"][df["metric_name"] == "b"]["sem"].item()),
            0.5,
        )

    def test_repr(self) -> None:
        self.assertEqual(
            str(Data(df=self.df)),
            REPR_1000,
        )
        with patch(f"{Data.__module__}.DF_REPR_MAX_LENGTH", 500):
            self.assertEqual(str(Data(df=self.df)), REPR_500)

    def test_clone(self) -> None:
        data = Data(df=self.df, description="test")
        data._db_id = 1234
        data_clone = data.clone()
        # Check equality of the objects.
        self.assertTrue(data.df.equals(data_clone.df))
        self.assertEqual(data.description, data_clone.description)
        # Make sure it's not the original object or df.
        self.assertIsNot(data, data_clone)
        self.assertIsNot(data.df, data_clone.df)
        self.assertIsNone(data_clone._db_id)

    def test_BadData(self) -> None:
        df = pd.DataFrame([{"bad_field": "0_0", "bad_field_2": {"x": 0, "y": "a"}}])
        with self.assertRaises(ValueError):
            Data(df=df)

    def test_EmptyData(self) -> None:
        df = Data().df
        self.assertTrue(df.empty)
        self.assertTrue(set(df.columns == Data.REQUIRED_COLUMNS))
        self.assertTrue(Data.from_multiple_data([]).df.empty)

    def test_CustomData(self) -> None:
        CustomData = custom_data_class(
            column_data_types={"metadata": str, "created_time": pd.Timestamp},
            required_columns={"metadata"},
        )
        data_entry = {
            "trial_index": 0,
            "arm_name": "0_1",
            "mean": 3.7,
            "sem": 0.5,
            "metric_name": "b",
            "metadata": "42",
            "created_time": "2018-09-20",
        }
        data = CustomData(df=pd.DataFrame([data_entry]))
        self.assertNotEqual(data, Data(self.df))
        self.assertTrue(isinstance(data.df.iloc[0]["created_time"], pd.Timestamp))

        data_entry2 = {
            "trial_index": 0,
            "arm_name": "0_1",
            "mean": 3.7,
            "sem": 0.5,
            "metric_name": "b",
            "created_time": "2018-09-20",
        }

        # Test without required column
        with self.assertRaises(ValueError):
            CustomData(df=pd.DataFrame([data_entry2]))

        # Try making regular data with extra column
        with self.assertRaises(ValueError):
            Data(df=pd.DataFrame([data_entry2]))

    def test_FromEvaluationsIsoFormat(self) -> None:
        now = pd.Timestamp.now()
        day = now.day
        for sem in (0.5, None):
            eval1 = (3.7, sem) if sem is not None else 3.7
            data = Data.from_evaluations(
                evaluations={"0_1": {"b": eval1}},
                trial_index=0,
                sample_sizes={"0_1": 2},
                start_time=now.isoformat(),
                end_time=now.isoformat(),
            )
            self.assertEqual(data.df["sem"].isnull()[0], sem is None)
            self.assertEqual(len(data.df), 1)
            self.assertNotEqual(data, Data(self.df))
            self.assertEqual(data.df["start_time"][0].day, day)
            self.assertEqual(data.df["end_time"][0].day, day)

    def test_FromEvaluationsMillisecondFormat(self) -> None:
        now_ms = current_timestamp_in_millis()
        day = pd.Timestamp(now_ms, unit="ms").day
        for sem in (0.5, None):
            eval1 = (3.7, sem) if sem is not None else 3.7
            data = Data.from_evaluations(
                evaluations={"0_1": {"b": eval1}},
                trial_index=0,
                sample_sizes={"0_1": 2},
                start_time=now_ms,
                end_time=now_ms,
            )
            self.assertEqual(data.df["sem"].isnull()[0], sem is None)
            self.assertEqual(len(data.df), 1)
            self.assertNotEqual(data, Data(self.df))
            self.assertEqual(data.df["start_time"][0].day, day)
            self.assertEqual(data.df["end_time"][0].day, day)

    def test_CloneWithoutMetrics(self) -> None:
        data = Data(df=self.df)
        expected = Data(
            df=pd.DataFrame(
                [
                    {
                        "arm_name": "0_0",
                        "mean": 1.8,
                        "sem": 0.3,
                        "trial_index": 1,
                        "metric_name": "b",
                        "start_time": "2018-01-01",
                        "end_time": "2018-01-02",
                    },
                    {
                        "arm_name": "0_1",
                        "mean": 3.7,
                        "sem": 0.5,
                        "trial_index": 1,
                        "metric_name": "b",
                        "start_time": "2018-01-01",
                        "end_time": "2018-01-02",
                    },
                    {
                        "arm_name": "0_2",
                        "mean": 3.0,
                        "sem": None,
                        "trial_index": 1,
                        "metric_name": "b",
                        "start_time": "2018-01-01",
                        "end_time": "2018-01-02",
                    },
                ]
            )
        )
        self.assertEqual(clone_without_metrics(data, {"a"}), expected)

    def test_from_multiple(self) -> None:
        with self.subTest("Combinining non-empty Data"):
            data = Data.from_multiple_data([Data(df=self.df), Data(df=self.df)])
            self.assertEqual(len(data.df), 2 * len(self.df))

        with self.subTest("Combining empty Data makes empty Data"):
            data = Data.from_multiple_data([Data(), Data()])
            self.assertEqual(data, Data())

        with self.subTest("Can't combine different types"):
            CustomData = custom_data_class()
            with self.assertRaisesRegex(
                TypeError, "All data objects must be instances of"
            ):
                CustomData.from_multiple_data([Data(), CustomData()])

    def test_FromMultipleDataMismatchedTypes(self) -> None:
        # create two custom data types
        CustomDataA = custom_data_class(
            column_data_types={"metadata": str, "created_time": pd.Timestamp},
            required_columns={"metadata"},
        )
        CustomDataB = custom_data_class(column_data_types={"year": pd.Timestamp})

        # Test using `Data.from_multiple_data` to combine non-Data types
        with self.assertRaisesRegex(TypeError, "All data objects must be instances of"):
            Data.from_multiple_data([CustomDataA(), CustomDataB()])

        # Test data of multiple non-empty types raises a value error
        with self.assertRaises(ValueError):
            data_elt_A = CustomDataA(
                df=pd.DataFrame(
                    [
                        {
                            "arm_name": "0_1",
                            "mean": 3.7,
                            "sem": 0.5,
                            "metric_name": "b",
                            "metadata": "42",
                            "created_time": "2018-09-20",
                        }
                    ]
                )
            )
            data_elt_B = CustomDataB(
                df=pd.DataFrame(
                    [
                        {
                            "arm_name": "0_1",
                            "mean": 3.7,
                            "sem": 0.5,
                            "metric_name": "b",
                            "year": "2018-09-20",
                        }
                    ]
                )
            )
            Data.from_multiple_data([data_elt_A, data_elt_B])

    def test_from_multiple_with_generator(self) -> None:
        data = Data.from_multiple_data(Data(df=self.df) for _ in range(2))
        self.assertEqual(len(data.df), 2 * len(self.df))

    def test_GetFilteredResults(self) -> None:
        data = Data(df=self.df)
        # pyre-fixme[6]: For 1st param expected `Dict[str, typing.Any]` but got `str`.
        # pyre-fixme[6]: For 2nd param expected `Dict[str, typing.Any]` but got `str`.
        actual_filtered = data.get_filtered_results(arm_name="0_0", metric_name="a")
        # Create new Data to replicate timestamp casting.
        expected_filtered = Data(
            pd.DataFrame(
                [
                    {
                        "arm_name": "0_0",
                        "metric_name": "a",
                        "mean": 2.0,
                        "sem": 0.2,
                        "trial_index": 1,
                        "start_time": "2018-01-01",
                        "end_time": "2018-01-02",
                    },
                ]
            )
        ).df
        self.assertTrue(actual_filtered.equals(expected_filtered))

    def test_data_column_data_types_default(self) -> None:
        self.assertEqual(Data.column_data_types(), Data.COLUMN_DATA_TYPES)

    def test_data_column_data_types_with_extra_columns(self) -> None:
        bartype = random.choice([str, int, float])
        columns = Data.column_data_types(extra_column_types={"foo": bartype})
        for c, t in Data.COLUMN_DATA_TYPES.items():
            self.assertEqual(columns[c], t)
        self.assertEqual(columns["foo"], bartype)

    def test_data_column_data_types_with_removed_columns(self) -> None:
        columns = Data.column_data_types(excluded_columns=["fidelities"])
        self.assertNotIn("fidelities", columns)
        for c, t in Data.COLUMN_DATA_TYPES.items():
            if c != "fidelities":
                self.assertEqual(columns[c], t)

    # there isn't really a point in doing this
    # this test just documents expected behavior
    # that excluded_columns wins out
    def test_data_column_data_types_with_extra_columns_also_deleted(self) -> None:
        bartype = random.choice([str, int, float])
        excluded_columns = ["fidelities", "foo"]
        columns = Data.column_data_types(
            extra_column_types={"foo": bartype},
            excluded_columns=excluded_columns,
        )
        self.assertNotIn("fidelities", columns)
        self.assertNotIn("foo", columns)
        for c, t in Data.COLUMN_DATA_TYPES.items():
            if c not in excluded_columns:
                self.assertEqual(columns[c], t)
