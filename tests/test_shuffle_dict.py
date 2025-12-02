
from spacenet.dataset.data_processing import shuffle_dict_lists


d = {
    "a": [1, 2, 3],
    "b": ["x", "y", "z"],
    "c": [True, False, True]
}

print(shuffle_dict_lists(d))