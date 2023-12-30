import copy
from multiprocessing import Queue, Process
import cv2 as cv
import numpy as np
import mediapipe as mp
import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode

from utils import CvFpsCalc
from main import draw_landmarks, draw_stick_figure

from fake_objects import FakeResultObject, FakeLandmarksObject, FakeLandmarkObject
from functions import calculate_angle
from turn import get_ice_servers


_SENTINEL_ = "_SENTINEL_"



def pose_process(
    in_queue: Queue,
    out_queue: Queue,
):
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    while True:
        input_item = in_queue.get(timeout=10)
        if isinstance(input_item, type(_SENTINEL_)) and input_item == _SENTINEL_:
            break

        results = pose.process(input_item)
        picklable_results = FakeResultObject(pose_landmarks=FakeLandmarksObject(landmark=[
            FakeLandmarkObject(
                x=pose_landmark.x,
                y=pose_landmark.y,
                z=pose_landmark.z,
                visibility=pose_landmark.visibility,
            ) for pose_landmark in results.pose_landmarks.landmark
        ]))
        out_queue.put_nowait(picklable_results)


class Tokyo2020PictogramVideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self._in_queue = Queue()
        self._out_queue = Queue()
        self._pose_process = Process(target=pose_process, kwargs={
            "in_queue": self._in_queue,
            "out_queue": self._out_queue,
        })
        self._cvFpsCalc = CvFpsCalc(buffer_len=10)

        self._pose_process.start()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # display_fps = self._cvFpsCalc.get()

        image = frame.to_ndarray(format="bgr24")
        image = cv.flip(image, 1)  # Mirror display
        image = copy.deepcopy(image) #just copies the image so the original image will stay unaffected
        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mp_pose = mp.solutions.pose

        results = self._infer_pose(image)

        if results.pose_landmarks is not None:
            landmarks1 = results.pose_landmarks.landmark
            shoulder1 = [int(landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image.shape[1]),
                        int(landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image.shape[0])]
            elbow1 = [int(landmarks1[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image.shape[1]),
                    int(landmarks1[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image.shape[0])]
            wrist1 = [int(landmarks1[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image.shape[1]),
                    int(landmarks1[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image.shape[0])]
            hip1 = [int(landmarks1[mp_pose.PoseLandmark.LEFT_HIP.value].x * image.shape[1]),
                    int(landmarks1[mp_pose.PoseLandmark.LEFT_HIP.value].y * image.shape[0])]

            angle1 = calculate_angle(shoulder1, elbow1, wrist1)
            hip_angle = calculate_angle(shoulder1, hip1, elbow1)


            normal_range = (20,160)

            if not (normal_range[0] <= hip_angle <= normal_range[1]):
                cv.putText(image, "ELBOW TOO FAR FROM HIP", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            cv.putText(image, f'Angle: {round(angle1, 2)}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.putText(image, 'REPS:', (300,50), 
                        cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # cv.putText(image, str(counter), (450,50), 
            #     cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            cv.circle(image, tuple(shoulder1), 10, (255, 255, 255), -1)
            cv.circle(image, tuple(elbow1), 10, (255, 255, 255), -1)
            cv.circle(image, tuple(wrist1), 10, (255, 255, 255), -1)
            cv.circle(image, tuple(hip1), 10, (255, 255, 255), -1)


            if 32 <= abs(angle1) <= 178:
                cv.putText(image, 'YES', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif 32 > abs(angle1):
                cv.putText(image, 'ARM TOO CLOSE, ELBOW TENSION', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv.putText(image, 'OVEREXTENDING', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            

            # image = draw_landmarks(image, results.pose_landmarks)

        return av.VideoFrame.from_ndarray(image, format="rgb24")

    def _infer_pose(self, image):
        self._in_queue.put_nowait(image)
        return self._out_queue.get(timeout=10)

    def _stop_pose_process(self):
        self._in_queue.put_nowait(_SENTINEL_)
        self._pose_process.join(timeout=10)

    def __del__(self):
        self._stop_pose_process()


def main():
    def processor_factory():
        return Tokyo2020PictogramVideoProcessor()

    webrtc_ctx = webrtc_streamer(
        key="tokyo2020-Pictogram",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=processor_factory,
    )
    st.session_state["started"] = webrtc_ctx.state.playing


if __name__ == "__main__":
    main()