"""
Copyright 2021 Semiotic AI, Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Test the cadr.utils module"""

import cadr.utils as utils


def test_list_to_dict():
    value = [{"foo": [0, 1], "bar": [2, 3]}, {"foo": [2, 3], "bar": [0, 1]}]
    converted = utils.list_to_dict(value)

    assert converted["foo"] == [[0, 1], [2, 3]]
    assert converted["bar"] == [[2, 3], [0, 1]]
