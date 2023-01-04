"""
Utilities for file paths. Must exist at the root so it is importable.
References:
https://github.com/google/ldif/blob/master/ldif/util/path_util.py
"""

import os


def get_path_to_ldif_root():
    """Finds the LDIF root directory by assuming the script is at the root."""
    expected_util = os.path.dirname(os.path.abspath(__file__))
    # Sanity check that the script lives in the root directory:
    if expected_util.split("/")[-1] != "util":
        raise ValueError(
            "Error: Script is not located in the LDIF package util folder, or the"
            f" util folder been renamed. Detected {os.path.abspath(__file__)}."
        )
    ldif_root = "/".join(expected_util.split("/")[:-1])
    if ldif_root.split("/")[-1] != "ldif":
        raise ValueError(
            "Error: Util folder is no longer located in the LDIF root. Please"
            " update util/path_util.py"
        )
    return ldif_root


def get_path_to_ldif_parent():
    ldif_root = get_path_to_ldif_root()
    return os.path.dirname(ldif_root)


def package_to_abs(path):
    root = get_path_to_ldif_root()
    return os.path.join(root, path)


def gaps_path():
    return package_to_abs("gaps/bin/x86_64/")


def test_data_path():
    return package_to_abs("test_data/")


def util_test_data_path():
    return package_to_abs("util/test_data/")


def create_test_output_dir():
    path = package_to_abs("test_outputs/")
    if not os.path.isdir(path):
        os.mkdir(path)
    return path
