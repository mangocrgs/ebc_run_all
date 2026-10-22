"""Quick runtime check, also runnable through the frozen app's --stage option."""
import sys
import numpy as np
import mediapipe as mp
import google.protobuf
from packaging.specifiers import SpecifierSet


def main():
    version = google.protobuf.__version__
    if version not in SpecifierSet('>=4.25.3,<5'):
        raise RuntimeError('MediaPipe 0.10.14 requires protobuf>=4.25.3,<5; found ' + version)
    for static in (True, False):
        with mp.solutions.face_mesh.FaceMesh(
                static_image_mode=static, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.4, min_tracking_confidence=0.4) as mesh:
            result = mesh.process(np.zeros((256, 256, 3), dtype=np.uint8))
            assert not result.multi_face_landmarks
    print('PASS: FaceMesh initialization and frame processing in both tracking modes', flush=True)
    print('MediaPipe %s; Protobuf %s; frozen=%s' %
          (mp.__version__, version, getattr(sys, 'frozen', False)), flush=True)


if __name__ == '__main__':
    main()
