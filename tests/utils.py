import os


_PRINT_END = "\t---->\t"

_TEST_PATH = os.path.dirname(
    __file__
)  # /Users/sameergururajmathad/se-489/team_ops/tests

# ------ Data -------
_PARENT_PATH = os.path.dirname(_TEST_PATH)  # /Users/sameergururajmathad/se-489/team_ops
_DATA_PATH = os.path.join(
    _PARENT_PATH, "data"
)  # /Users/sameergururajmathad/se-489/team_ops/data

_RAW_PATH = os.path.join(_DATA_PATH, "raw")
_PROCESSED_PATH = os.path.join(_DATA_PATH, "processed")


# ------- Training ------
_TRAIN_PATH = os.path.join(_PARENT_PATH, "team_ops/models")
_SAVE_PATH = os.path.join(_PARENT_PATH, "models")
