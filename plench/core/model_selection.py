import itertools
import numpy as np

def filter_step0_records(records):
    """Filter step 0"""
    return records.filter(lambda r: r['step'] != 0)

class SelectionMethod:
    """Abstract class whose subclasses implement strategies for model
    selection across hparams and timesteps."""

    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(self, run_records):
        """
        Given records from a run, return a {val_acc, test_acc} dict representing
        the best val-acc and corresponding test-acc for that run.
        """
        raise NotImplementedError

    @classmethod
    def hparams_accs(self, records):
        """
        Given all records from a single (dataset, algorithm) pair,
        return a sorted list of (run_acc, records) tuples.
        """
        return (records.group('args.hparams_seed')
            .map(lambda _, run_records:
                (
                    self.run_acc(run_records),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]['val_acc'])[::-1]
        )

    @classmethod
    def sweep_acc(self, records):
        """
        Given all records from a single (dataset, algorithm) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc']
        else:
            return None

class OracleSelectionMethod(SelectionMethod):
    """Like Selection method which picks argmax(val_accuracy) across all hparams
    and checkpoints, but instead of taking the argmax over all
    checkpoints, we pick the last checkpoint, i.e. no early stopping."""
    name = "Oracle Accuracy"

    @classmethod
    def run_acc(self, run_records):
        run_records = filter_step0_records(run_records)
        if not len(run_records):
            return None
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]
        return {
            'val_acc':  chosen_record['val_accuracy'],
            'test_acc': chosen_record['test_acc']
        }

class OracleSelectionWithEarlyStoppingMethod(SelectionMethod):
    """Picks argmax(val_accuracy), with early stopping"""
    name = "Oracle Accuracy w/ ES"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""

        return {
            'val_acc':  record['val_accuracy'],
            'test_acc': record['test_acc']
        }

    @classmethod
    def run_acc(self, run_records):
        test_records = filter_step0_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')

class CoveringRateSelectionMethod(SelectionMethod):
    """Picks argmax(val_covering_rate)"""
    name = "Covering Rate"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""

        return {
            'val_acc':  record['val_covering_rate'],
            'test_acc': record['test_acc']
        }

    @classmethod
    def run_acc(self, run_records):
        test_records = filter_step0_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')

class ApproximatedAccuracySelectionMethod(SelectionMethod):
    """Picks argmax(val_approximated_acc)"""
    name = "Approximated Accuracy"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""

        return {
            'val_acc':  record['val_approximated_acc'],
            'test_acc': record['test_acc']
        }

    @classmethod
    def run_acc(self, run_records):
        test_records = filter_step0_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')
