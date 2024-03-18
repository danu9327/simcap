import numpy as np
import torch

# CV
import cv2
import supervision as sv

# YOLOv9
from models.common import DetectMultiBackend, AutoShape
from utils.torch_utils import select_device
from utils.general import set_logging
from supervision import Detections as BaseDetections
from supervision.config import CLASS_NAME_DATA_FIELD
import numpy as np
import torch

class ExtendedDetections(BaseDetections):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.centroids = []  # centroids 속성 초기화

    @classmethod
    def from_yolov9(cls, yolov9_results) -> 'ExtendedDetections':
        xyxy, confidences, class_ids, centroids = [], [], [], []

        for det in yolov9_results.pred:
            for *xyxy_coords, conf, cls_id in reversed(det):
                xyxy.append(torch.stack(xyxy_coords).cpu().numpy())
                confidences.append(float(conf))
                class_ids.append(int(cls_id))
                center = ((xyxy_coords[0] + xyxy_coords[2]) / 2, (xyxy_coords[1] + xyxy_coords[3]) / 2)
                centroids.append(center)  # 중심점 저장

        class_names = np.array([yolov9_results.names[i] for i in class_ids])

        if not xyxy:
            return cls.empty()

        detections = cls(
            xyxy=np.vstack(xyxy),
            confidence=np.array(confidences),
            class_id=np.array(class_ids),
            data={CLASS_NAME_DATA_FIELD: class_names},
        )
        detections.centroids = centroids  # 중심점 데이터 추가
        return detections
SOURCE_VIDEO_PATH = "input.mp4"
TARGET_VIDEO_PATH = "output.mp4"
#(best.pt파일 준비된 상황이면)
def prepare_yolov9(model, conf=0.4, iou=0.7, classes=None, agnostic_nms=False, max_det=1000):
    model.conf = conf
    model.iou = iou
    model.classes = classes
    model.agnostic = agnostic_nms
    model.max_det = max_det
    return model
import supervision as sv
#(best.pt파일 준비된 상황이면)
def setup_model_and_video_info(model, config, source_path):
    # Initialize and configure YOLOv9 model
    model = prepare_yolov9(model, **config)
    
    if source_path is None:
        # 웹캠 스트림을 처리하는 경우, 직접 VideoInfo 객체를 생성
        # 이 예에서는 대략적인 값을 사용합니다. 실제로는 웹캠의 속성을 조회하여 설정해야 합니다.
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        cap.release()  # VideoCapture 객체는 여기에서만 필요하므로 바로 해제
        
        video_info = sv.VideoInfo(fps=fps, width=width, height=height, total_frames=0)
    else:
        # 비디오 파일 경로가 제공된 경우, 기존 로직을 사용하여 VideoInfo 생성
        video_info = sv.VideoInfo.from_video_path(source_path)
    
    return model, video_info


def create_byte_tracker(video_info):
    # Setup BYTETracker with video information
    return sv.ByteTrack(track_thresh=0.25, track_buffer=250, match_thresh=0.95, frame_rate=video_info.fps)

def setup_annotators():
    # Initialize various annotators for bounding boxes, traces, and labels
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    round_box_annotator = sv.RoundBoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    corner_annotator = sv.BoxCornerAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50, color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, color_lookup=sv.ColorLookup.TRACK)
    dot_annotator = sv.DotAnnotator()  # DotAnnotator 추가
    return [bounding_box_annotator, round_box_annotator, corner_annotator], trace_annotator, label_annotator, dot_annotator

def setup_counting_zone(counting_zone, video_info):
    # Configure counting zone based on provided parameters
    if counting_zone == 'whole_frame':
        polygon = np.array([[0, 0], [video_info.width-1, 0], [video_info.width-1, video_info.height-1], [0, video_info.height-1]])
    else:
        polygon = np.array(counting_zone)
    polygon_zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=(video_info.width, video_info.height), triggering_position=sv.Position.CENTER)
    polygon_zone_annotator = sv.PolygonZoneAnnotator(polygon_zone, sv.Color.ROBOFLOW, thickness=2*(2 if counting_zone=='whole_frame' else 1), text_thickness=1, text_scale=0.5)
    return polygon_zone, polygon_zone_annotator


def annotate_frame(frame, index, video_info, detections, byte_tracker, counting_zone, polygon_zone, polygon_zone_annotator, trace_annotator, annotators_list, label_annotator, show_labels, model, dot_annotator):
    if index is None:
        # 여기서는 예시로 0을 사용하지만, 실제 로직에 맞게 적절히 조정해야 합니다.
        index = 0
    section_index = int(index / (video_info.total_frames / len(annotators_list)) if video_info.total_frames else 1)    
    detections = byte_tracker.update_with_detections(detections)
    annotated_frame = frame.copy()
    if counting_zone is not None:
        is_inside_polygon = polygon_zone.trigger(detections)
        detections = detections[is_inside_polygon]
        annotated_frame = polygon_zone_annotator.annotate(annotated_frame)

    # Annotate frame with traces
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)

    # Annotate frame with various bounding boxes
    #section_index = int(index / (video_info.total_frames / len(annotators_list)))
    annotated_frame = annotators_list[section_index].annotate(scene=annotated_frame, detections=detections)

    # Optionally, add labels to the annotations
    if show_labels:
        annotated_frame = add_labels_to_frame(label_annotator, annotated_frame, detections, model)
    annotated_frame = dot_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )

    frame_rgb = frame[..., ::-1]
    results = model(frame_rgb, size=640, augment=False)
    detection = ExtendedDetections.from_yolov9(results)
    triggering_position=sv.Position.CENTER
    for id, center in zip(detections.tracker_id, detection.centroids):
        center_int = (int(center[0]), int(center[1]))
        cv2.circle(frame, center_int, radius=5, color=(0, 0, 255), thickness=-1)
        print("성공이다식별아디:",id, center_int)  # 중심점 좌표 출력
    print("-------------") # 쉬어가기

    return annotated_frame

def add_labels_to_frame(annotator, frame, detections, model):
    # 각 검출된 객체에 대해 tracker_id만을 라벨로 생성
    labels = [f"ID: {tracker_id}" for tracker_id in detections.tracker_id]
    # 수정된 라벨 리스트를 사용하여 프레임에 주석 추가
    return annotator.annotate(scene=frame, detections=detections, labels=labels)


def process_video(model, config=dict(conf=0.1, iou=0.45, classes=None,), counting_zone=None, show_labels=False, source_path=SOURCE_VIDEO_PATH, target_path=TARGET_VIDEO_PATH):
    model, video_info = setup_model_and_video_info(model, config, source_path)
    byte_tracker = create_byte_tracker(video_info)
    annotators_list, trace_annotator, label_annotator, dot_annotator = setup_annotators()
    polygon_zone, polygon_zone_annotator = setup_counting_zone(counting_zone, video_info) if counting_zone else (None, None)

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        frame_rgb = frame[..., ::-1]  # Convert BGR to RGB
        results = model(frame_rgb, size=608, augment=False)
        detections = ExtendedDetections.from_yolov9(results)
        return annotate_frame(frame, index, video_info, detections, byte_tracker, counting_zone, polygon_zone, polygon_zone_annotator, trace_annotator, annotators_list, label_annotator, show_labels, model, dot_annotator)

    sv.process_video(source_path=source_path, target_path=target_path, callback=callback)    