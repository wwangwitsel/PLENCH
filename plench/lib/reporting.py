import collections

import json
import os

import tqdm

from .query import Q

def load_records(path):
    records = []
    for i, subdir in tqdm.tqdm(list(enumerate(os.listdir(path))),
                               ncols=80,
                               leave=False):
        results_path = os.path.join(path, subdir, "results.jsonl")
        try:
            with open(results_path, "r") as f:
                for line in f:
                    records.append(json.loads(line[:-1]))
        except IOError:
            pass

    return Q(records)

def get_grouped_records(records):
    """Group records by (trial_seed, dataset, algorithm). """
    result = collections.defaultdict(lambda: [])
    for r in records:
        group = (r["args"]["trial_seed"],
            r["args"]["dataset"],
            r["args"]["algorithm"])
        result[group].append(r)
    return Q([{"trial_seed": t, "dataset": d, "algorithm": a,
        "records": Q(r)} for (t,d,a),r in result.items()])
