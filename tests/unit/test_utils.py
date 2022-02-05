"""Test the cadr.utils module"""

import cadr.utils as utils


def test_list_to_dict():
    value = [{"foo": [0, 1], "bar": [2, 3]}, {"foo": [2, 3], "bar": [0, 1]}]
    converted = utils.list_to_dict(value)

    assert converted["foo"] == [[0, 1], [2, 3]]
    assert converted["bar"] == [[2, 3], [0, 1]]
