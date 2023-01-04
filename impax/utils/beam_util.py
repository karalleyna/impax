"""
Utilities for apache beam jobs.
References:
https://github.com/google/ldif/blob/master/ldif/util/beam_util.py
"""

import apache_beam as beam


def filter_success(elt):
    return [] if "FAILED" in elt else [elt]


def filter_failure(elt):
    return [elt] if "FAILED" in elt else []


def report_errors(elts):
    errs = []
    for elt in elts:
        errs.append(f"{elt['mesh_identifier']}|||{elt['FAILED']}\n")
    return "\n".join(errs) + "\n"


def map_and_report_failures(inputs, f, name, fail_base, applier=None):
    """Applies a function and then parses out the successes and failures."""
    if applier is None:
        applier = beam.Map
    mapped = inputs | name >> applier(f)
    failed_to_map = mapped | f"Get{name}Failed" >> beam.FlatMap(filter_failure)
    successfully_mapped = mapped | f"Get{name}Succeeded" >> beam.FlatMap(filter_success)
    fail_file = fail_base + f"_failed_to_{name}.txt"
    _ = (
        failed_to_map
        | f"CombineErrPcollFor{name}" >> beam.combiners.ToList()
        | f"MakeErrStringFor{name}" >> beam.Map(report_errors)
        | f"WritErrStringFor{name}"
        >> beam.io.WriteToText(fail_file, num_shards=1, shard_name_template="")
    )
    return successfully_mapped
