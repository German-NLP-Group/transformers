# Copyright 2021 Philip May
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import unittest
from unittest.mock import patch

from transformers.integrations import MLflowCallback


class MLflowCallbackTest(unittest.TestCase):

    @patch("transformers.TrainerState")
    @patch("transformers.TrainingArguments")
    def test_mlflow_callback(self, args, state):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["MLFLOW_TRACKING_URI"] = f"file:{tmpdir}"

            mlflow_log_key = "some_key"
            mlflow_log_value = "some_value"
            args.to_dict.return_value({mlflow_log_key: mlflow_log_value})

            state.is_world_process_zero = True

            mlflow_callback = MLflowCallback()
            mlflow_callback.setup(args, state, model=None)

            args.to_dict.assert_called_once()
            mlflow_callback.on_train_begin(args, state, control=None, model=None)

            state.global_step = 0
            metrics_0 = {"auc": 0.75, "acc": 0.79}
            mlflow_callback.on_log(args, state, control=None, logs=metrics_0, model=None)

            state.global_step = 1
            metrics_1 = {"auc": 0.77, "acc": 0.82}
            mlflow_callback.on_log(args, state, control=None, logs=metrics_1, model=None)

            mlflow_callback.on_train_end(args, state, control=None)
