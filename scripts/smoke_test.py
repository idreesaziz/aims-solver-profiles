"""Exercise the fitted predictor without network or external credentials."""
import json
import math
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def blocked(*args, **kwargs):
    raise AssertionError('The public predictor must not access the network.')


socket.socket.connect = blocked
socket.create_connection = blocked
import model


def subject(name):
    return dict(normalized_name=name, provider='Test', release_date='', access_date='',
                harness='test', harness_version='', reasoning_effort='', subject_features_extra='')


def item(number, benchmark='example'):
    return dict(item_content=f'Synthetic arithmetic {number}: What is 3+4?',
                item_features='', interactors='', benchmark_id=benchmark)


a, b = subject('solver-a'), subject('solver-b')
query = [a, item(40)]
labels = [[[a, item(0)], 1], [[b, item(0)], 0], [[b, item(1)], 1]]
before = model.predict(query, [])
after = model.predict(query, labels)
assert all(type(p) is float and math.isfinite(p) and 0 <= p <= 1 for p in [before, after])
assert after == model.predict(query, labels)
assert after == model.predict(query, labels + [[[a, item(2, 'other-benchmark')], 0]])
print(json.dumps(dict(passed=True, prior=before, adapted=after, network_blocked=True,
                      deterministic=True, benchmark_isolation=True)))
