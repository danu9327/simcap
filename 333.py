import cv2
import numpy as np
import torch
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import set_logging
import track
import supervision as sv
from models.common import DetectMultiBackend, AutoShape
# 가정: track.py의 나머지 부분은 이미 임포트되었거나 같은 파일에 포함되어 있음

# 모델 및 비디오 정보 설정 함수가 정의되어 있다고 가정합니다.
# setup_model_and_video_info, create_byte_tracker, setup_annotators, setup_counting_zone 등

def process_webcam(model, config=dict(conf=0.1, iou=0.45, classes=None), counting_zone=None, show_labels=False): #'whole_frame', [[100, 50], [400, 50], [400, 250], [100, 250]] 이렇게 지정해서 counting zone을 설정도 가능해여!
    # 웹캠을 위한 준비. 여기서는 실제 video_info 대신 간단한 설정을 사용합니다.
    cap = cv2.VideoCapture(0)  # 웹캠 열기
    ret, frame = cap.read()
    if not ret:
        print("웹캠을 시작할 수 없습니다.")
        return
    
    video_info = sv.VideoInfo(fps=30,  # 대략적인 웹캠 FPS
                              width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                              total_frames=0)  # 실시간 스트림이므로 전체 프레임 수는 0

    model, _ = track.setup_model_and_video_info(model, config, None)  # source_path는 None으로 설정
    byte_tracker = track.create_byte_tracker(video_info)
    annotators_list, trace_annotator, label_annotator, dot_annotator = track.setup_annotators()
    polygon_zone, polygon_zone_annotator = track.setup_counting_zone(counting_zone, video_info) if counting_zone else (None, None)
    def callback(frame, index):
        frame_rgb = frame[..., ::-1]  # Convert BGR to RGB
        results = model(frame_rgb, size=608, augment=False)
        detections = track.ExtendedDetections.from_yolov9(results)
        return track.annotate_frame(frame, index, video_info, detections, byte_tracker, counting_zone, polygon_zone, polygon_zone_annotator, trace_annotator, annotators_list, label_annotator, show_labels, model, dot_annotator)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 여기서 callback 함수와 유사한 처리를 수행
        annotated_frame = callback(frame, None)  # index는 실시간 처리에서는 사용되지 않음
        
        # 처리된 프레임을 표시
        cv2.imshow('Processed Webcam Feed', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'를 누르면 종료
            break
    
    # 종료 처리
    cap.release()
    cv2.destroyAllWindows()


# 모델 로딩 및 설정
model_path = 'best_555.pt'  # 모델 경로 설정
device = select_device('cpu')  # 'cpu'나 'cuda:0' 같은 디바이스 설정
model = DetectMultiBackend(model_path, device=device, dnn=False)
model = AutoShape(model)  # 모델을 AutoShape으로 감싸서 사용하기 쉽게 만듦

# 웹캠 프로세스 시작
process_webcam(model)

