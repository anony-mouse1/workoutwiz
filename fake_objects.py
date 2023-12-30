"""
Picklable objects to imitate the result object of pose estimation
so that the results can be passed over processes through multiprocessing.Pipe.
"""

from typing import NamedTuple, List


class FakeLandmarkObject(NamedTuple):
    x: float
    y: float
    z: float
    visibility: bool

class FakeLandmarksObject(NamedTuple):
    landmark: List[FakeLandmarkObject]


class FakeResultObject(NamedTuple):
    pose_landmarks: FakeLandmarksObject
