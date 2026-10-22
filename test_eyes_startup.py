import json
import os
import sys
import tempfile
from unittest.mock import patch
import ebc_eyes as E


def test_eyes_main_loads_stimulus_and_reaches_trial_decode():
    # Exercise real geometry and main without decoding video or running FaceMesh.
    for role, anchor, color in [('baseline_us', 'cs', 'blue'),
                                 ('conditioning', 'cs', 'yellow'),
                                 ('conditioning', 'us', 'blue')]:
        with tempfile.TemporaryDirectory() as d:
            rec = dict(tag='sample', path='unused.mp4', role=role, anchor=anchor)
            cfg = dict(recordings=[rec], protocol=dict(cs_ms=400, us_onset_ms=350,
                       us_dur_ms=50), output_dir=d)
            with open(os.path.join(d, 'trials.json'), 'w') as f:
                json.dump(dict(trials=[dict(session='sample', anchor_frame=120,
                                           session_trial=1)]), f)
            with open(os.path.join(d, 'sample_stim.json'), 'w') as f:
                json.dump(dict(leds={color: dict(position=dict(x=100, y=100,
                          spread_x=0, spread_y=0))}), f)
            calls = []
            def decode(*args, **kwargs):
                calls.append((args, kwargs))
                return iter([])  # Short window deliberately skips tracking.
            with patch.object(sys, 'argv', ['ebc_eyes.py', 'config.json', 'sample']), \
                 patch.object(E.C, 'load', return_value=cfg), \
                 patch.object(E, 'work_dir', return_value=d), \
                 patch.object(E, 'probe', return_value=(640, 480, 120., 1000)), \
                 patch.object(E, 'face_box', return_value=dict(x0=200, x1=300,
                              y0=150, y1=300, n=26)), \
                 patch.object(E, 'frames', side_effect=decode), \
                 patch.dict(sys.modules, {'mediapipe': object(), 'cv2': object()}):
                E.main()
            assert len(calls) == 1, 'trial decoding was never reached'
            with open(os.path.join(d, 'sample_traces.json')) as f:
                assert json.load(f) == {}

if __name__ == '__main__':
    test_eyes_main_loads_stimulus_and_reaches_trial_decode()
    print('PASS: baseline US, conditioning CS, and inferred CS from US reach decoding')
