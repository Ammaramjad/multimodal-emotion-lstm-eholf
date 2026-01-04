"""
MediaPipe-based facial landmark extractor and geometric descriptor computation.
"""

import numpy as np
import cv2

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except Exception:
    _HAS_MEDIAPIPE = False

mp_face_mesh = mp.solutions.face_mesh if _HAS_MEDIAPIPE else None

MP_IDX = {
    'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133],
    'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263],
    'left_eyebrow': [70, 63, 105, 66, 107],
    'right_eyebrow': [336, 296, 334, 293, 300],
    'top_lip': [13, 14, 15],
    'bottom_lip': [17, 18, 19],
    'mouth_left': [61],
    'mouth_right': [291],
    'nose_tip': [1, 4, 5]
}

def extract_face_mesh_landmarks(frame_bgr, max_faces=1, static_image_mode=True, refine_landmarks=False):
    if not _HAS_MEDIAPIPE:
        raise RuntimeError("mediapipe not installed.")
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=static_image_mode,
                               max_num_faces=max_faces,
                               refine_landmarks=refine_landmarks,
                               min_detection_confidence=0.5) as fm:
        results = fm.process(rgb)
        if not results or not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0]
        arr = []
        for p in lm.landmark:
            arr.append([p.x * w, p.y * h, p.z * w])
        return np.asarray(arr, dtype=float)

def geometric_features_from_landmarks(landmarks):
    if landmarks is None:
        return np.zeros(10, dtype=float)
    def mean_points(idx_list):
        pts = np.asarray([landmarks[i][:2] for i in idx_list])
        return pts.mean(axis=0)
    top_lip = mean_points(MP_IDX['top_lip'])
    bottom_lip = mean_points(MP_IDX['bottom_lip'])
    mouth_open = np.linalg.norm(top_lip - bottom_lip)
    left_corner = mean_points(MP_IDX['mouth_left'])
    right_corner = mean_points(MP_IDX['mouth_right'])
    lip_width = np.linalg.norm(left_corner - right_corner) + 1e-8
    left_eye = np.asarray([landmarks[i][:2] for i in MP_IDX['left_eye']])
    right_eye = np.asarray([landmarks[i][:2] for i in MP_IDX['right_eye']])
    def eye_aperture(eye_pts):
        ys = eye_pts[:, 1]
        return ys.max() - ys.min()
    eye_ap = (eye_aperture(left_eye) + eye_aperture(right_eye)) / 2.0
    l_eb = np.asarray([landmarks[i][:2] for i in MP_IDX['left_eyebrow']])
    r_eb = np.asarray([landmarks[i][:2] for i in MP_IDX['right_eyebrow']])
    eyebrow_metric = (np.std(l_eb[:, 1]) + np.std(r_eb[:, 1])) / 2.0
    nose = mean_points(MP_IDX['nose_tip'])
    eye_center = np.vstack([left_eye.mean(axis=0), right_eye.mean(axis=0)]).mean(axis=0)
    nose_eye_y = nose[1] - eye_center[1]
    mouth_top_left = np.linalg.norm(top_lip - left_corner)
    mouth_top_right = np.linalg.norm(top_lip - right_corner)
    left_corner_nose = np.linalg.norm(left_corner - nose)
    right_corner_nose = np.linalg.norm(right_corner - nose)
    feat = np.array([mouth_open, lip_width, eye_ap, eyebrow_metric, nose_eye_y,
                     mouth_top_left, mouth_top_right, left_corner_nose, right_corner_nose,
                     lip_width / (eye_ap + 1e-6)], dtype=float)
    return feat

def frame_geometric_and_deep(frame_bgr, resnet_embedder=None):
    landmarks = extract_face_mesh_landmarks(frame_bgr)
    geom = geometric_features_from_landmarks(landmarks)
    if resnet_embedder is None:
        return geom
    deep = resnet_embedder.deep_embedding(frame_bgr)
    return np.concatenate([deep, geom])