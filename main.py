# ==========================================================
# IMPORTS
# ==========================================================
import cv2
import numpy as np
import depthai as dai
from pathlib import Path
from skimage.feature import hog
import joblib


# ==========================================================
# PATHS (Relative to script location)
# ==========================================================
SCRIPT_DIR = Path(__file__).parent
YOLO_MODEL_PATH = SCRIPT_DIR / "models" / "best.pt"
SVM_MODEL_PATH = SCRIPT_DIR / "models" / "hog_svm_turn_slope.pkl"


# ==========================================================
# LOAD SVM
# ==========================================================
def load_svm_model():
    """Load SVM model and label encoder"""
    if not SVM_MODEL_PATH.exists():
        raise FileNotFoundError(f"SVM model not found at {SVM_MODEL_PATH}")
    return joblib.load(SVM_MODEL_PATH)


# ==========================================================
# HOG FEATURE EXTRACTOR
# ==========================================================
def extract_hog(cv2_img):
    """Extract HOG features from image"""
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return features


# ==========================================================
# CLASSICAL SLOPE FALLBACK
# ==========================================================
def classify_hill_classical(cv2_crop):
    """Classify slope using classical image processing as fallback"""
    gray = cv2.cvtColor(cv2_crop, cv2.COLOR_BGR2GRAY)
    arr = 255 - gray

    h, w = arr.shape
    crop = arr[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    crop = cv2.resize(crop, (100, 100))

    left = crop[:, :50]
    right = crop[:, 50:]

    left_dark = np.sum(left)
    right_dark = np.sum(right)

    if right_dark > left_dark * 1.15:
        return "uphill"
    elif left_dark > right_dark * 1.15:
        return "downhill"
    else:
        return "unknown"


# ==========================================================
# FINAL CLASSIFIER (HOG + SVM)
# ==========================================================
def classify_turn_or_slope(crop, sign_type, svm, label_encoder):
    """Classify turn direction or slope using HOG+SVM with fallback"""
    feat = extract_hog(crop)

    pred_idx = svm.predict([feat])[0]
    conf = np.max(svm.predict_proba([feat])[0])

    pred = label_encoder.inverse_transform([pred_idx])[0]

    if sign_type == "turn":
        if pred in ["turnL", "turnR"] and conf > 0.6:
            return pred
        return "unknown"

    if sign_type == "slope":
        if pred in ["uphill", "downhill"] and conf > 0.6:
            return pred

        # fallback to classical
        return classify_hill_classical(crop)

    return "unknown"


# ==========================================================
# OAK-D + YOLOv8
# ==========================================================
class OAKCameraYOLOv8Custom:
    def __init__(self, model_path, conf_thresh=0.5):
        self.model_path = model_path
        self.conf_thresh = conf_thresh

        self.classes = [
            "barred", "crossWalk", "NoPassB", "NoPassE",
            "park", "priority", "stop", "straight",
            "tunnelB", "tunnelE", "turn", "slope"
        ]

        rng = np.random.default_rng(42)
        self.colors = rng.integers(0, 255, (len(self.classes), 3)).tolist()

        self.pipeline = self.create_pipeline()
        self.svm = None
        self.label_encoder = None

    def create_pipeline(self):
        """Create DepthAI pipeline for OAK-D camera"""
        pipeline = dai.Pipeline()

        cam = pipeline.create(dai.node.ColorCamera)
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("rgb")

        cam.setPreviewSize(640, 352)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(30)

        cam.preview.link(xout.input)
        return pipeline

    def load_models(self):
        """Load YOLO and SVM models"""
        from ultralytics import YOLO
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found at {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        self.svm, self.label_encoder = load_svm_model()
        print("✓ YOLOv8 model loaded")
        print("✓ SVM model loaded")

    def process_frame(self, frame):
        """Process frame and return detections"""
        detections = []
        results = self.model(frame, verbose=False)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.conf_thresh:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                cls_name = self.classes[cls_id]

                crop = frame[y1:y2, x1:x2]

                det = {
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf
                }

                if cls_name in ["turn", "slope"] and crop.size > 0:
                    det["direction"] = classify_turn_or_slope(
                        crop, cls_name, self.svm, self.label_encoder
                    )

                detections.append(det)

        return detections

    def draw(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            color = tuple(map(int, self.colors[d["class_id"]]))

            label = d["class_name"]
            if "direction" in d:
                label += f":{d['direction']}"
            label += f" {d['confidence']:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def run(self):
        """Main loop for camera processing"""
        device = dai.Device(self.pipeline)
        q = device.getOutputQueue("rgb", 4, False)

        self.load_models()
        print("\n🎥 OAK-D camera running...")
        print("Press 'q' to quit\n")

        while True:
            pkt = q.tryGet()
            if pkt is None:
                continue

            frame = pkt.getCvFrame()
            detections = self.process_frame(frame)
            frame = self.draw(frame, detections)

            cv2.imshow("Traffic Sign Detection - YOLOv8 + HOG + SVM", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


# ==========================================================
# MAIN
# ==========================================================
def main():
    """Main entry point"""
    print("=" * 60)
    print("Traffic Sign Detection System")
    print("OAK-D Camera + YOLOv8 + HOG + SVM")
    print("=" * 60)
    
    try:
        oak = OAKCameraYOLOv8Custom(YOLO_MODEL_PATH)
        oak.run()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure your models are in the 'models' directory:")
        print(f"  - {YOLO_MODEL_PATH}")
        print(f"  - {SVM_MODEL_PATH}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()