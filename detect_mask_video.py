# import the necessary packages
import argparse
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import time
import cv2

from centroid_tracker import CentroidTracker
from config_manager import load_config
from debug_utils import debug_log, is_debug_enabled
from serial_reader import SerialTemperatureReader, find_temperature_port
from db import init_db, log_entry, should_log
from snapshot import save_violation_snapshot

FACE_CONFIDENCE = 0.5
MASK_CONFIDENCE_THRESHOLD = 70.0
PRIMARY_SWITCH_MARGIN = 0.20

_last_primary_id = None
_frame_times = []

def parse_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("--temp-port", type=str, default=None,
		help="ESP32 serial port, for example COM3. If omitted, config.json is used.")
	ap.add_argument("--temp-baud", type=int, default=None,
		help="ESP32 serial baud rate.")
	return vars(ap.parse_args())

def format_temperature_result(objectTempC, serialStatus, feverThresholdC):
	if objectTempC is None:
		if "on hold" in serialStatus.lower():
			return ("Temp: serial on hold", "TEMP ON HOLD", False)
		if "serial monitor" in serialStatus.lower():
			return ("Temp: paste reading from Serial Monitor", "TEMP UNKNOWN", False)
		if "blocked" in serialStatus.lower():
			return ("Temp: close Arduino Serial Monitor", "TEMP UNKNOWN", False)
		if "connected" in serialStatus.lower():
			if "upload" in serialStatus.lower() or "waiting" in serialStatus.lower():
				return ("Temp: upload MLX90614 sketch to ESP32", "TEMP UNKNOWN", False)
			return ("Temp: waiting for ESP32 data", "TEMP UNKNOWN", False)
		return ("Temp: ESP32 not connected", "TEMP UNKNOWN", False)

	tempText = f"Temp: {objectTempC:.2f} C"
	if objectTempC >= feverThresholdC:
		return (f"{tempText} | HIGH", "HIGH TEMP", True)

	return (f"{tempText} | NORMAL", "TEMP OK", False)

def add_face_for_prediction(frame, startX, startY, endX, endY, faces, locs):
	face = frame[startY:endY, startX:endX]
	if face.size == 0:
		return

	face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
	face = cv2.resize(face, (224, 224))
	face = img_to_array(face)
	face = preprocess_input(face)

	faces.append(face)
	locs.append((startX, startY, endX, endY))

def classify_mask(pred, confidence_threshold=MASK_CONFIDENCE_THRESHOLD):
	(mask, withoutMask) = pred
	confidence = float(max(mask, withoutMask) * 100)
	if confidence < confidence_threshold:
		return "Unknown", confidence
	if mask > withoutMask:
		return "Mask", confidence
	return "No Mask", confidence


def is_violation(mask_label, temperature_c, fever_threshold_c):
	if mask_label in ("No Mask", "Unknown"):
		return True
	if temperature_c is not None and temperature_c >= fever_threshold_c:
		return True
	return False


def _clip_box(startX, startY, endX, endY, width, height):
	startX = max(0, int(startX))
	startY = max(0, int(startY))
	endX = min(width - 1, int(endX))
	endY = min(height - 1, int(endY))
	if endX <= startX or endY <= startY:
		return None
	return (startX, startY, endX, endY)


def detect_faces(frame, faceNet, faceCascade):
	(h, w) = frame.shape[:2]
	locs = []

	if faceNet is not None:
		blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
			(104.0, 177.0, 123.0))
		faceNet.setInput(blob)
		detections = faceNet.forward()

		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > FACE_CONFIDENCE:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				clipped = _clip_box(startX, startY, endX, endY, w, h)
				if clipped is not None:
					locs.append(clipped)

	if len(locs) == 0 and faceCascade is not None:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.equalizeHist(gray)
		fallbackFaces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=1.08,
			minNeighbors=5,
			minSize=(60, 60)
		)
		for (x, y, fw, fh) in fallbackFaces:
			clipped = _clip_box(x, y, x + fw, y + fh, w, h)
			if clipped is not None:
				locs.append(clipped)

	# #region agent log
	debug_log("detect_mask_video.py:detect_faces", "face detection complete",
		{"dnn_faces": len(locs) if faceNet is not None else 0,
		 "total_faces": len(locs), "used_haar_fallback": faceNet is None or len(locs) > 0},
		hypothesis_id="B")
	# #endregion
	return locs

def predict_masks(frame, locs, maskNet):
	faces = []
	validLocs = []

	for (startX, startY, endX, endY) in locs:
		face = frame[startY:endY, startX:endX]
		if face.size == 0:
			continue

		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		faces.append(face)
		validLocs.append((startX, startY, endX, endY))

	if len(faces) == 0:
		return [], np.array([])

	faces = np.array(faces, dtype="float32")
	preds = maskNet.predict(faces, batch_size=min(32, len(faces)), verbose=0)
	# #region agent log
	debug_log("detect_mask_video.py:predict_masks", "mask predictions complete",
		{"face_count": len(faces), "pred_shape": list(preds.shape)},
		hypothesis_id="D")
	# #endregion
	return validLocs, preds

def _centroid(rect):
	(startX, startY, endX, endY) = rect
	return (int((startX + endX) / 2.0), int((startY + endY) / 2.0))

def choose_primary_person(tracked, frameShape, sensorPoint=None, last_primary_id=None):
	global _last_primary_id
	if not tracked:
		_last_primary_id = None
		return None

	(h, w) = frameShape[:2]
	if sensorPoint is None:
		sensorPoint = (w // 2, h // 3)

	candidate = min(
		tracked.keys(),
		key=lambda objectID: np.linalg.norm(
			np.array(_centroid(tracked[objectID])) - np.array(sensorPoint)
		),
	)

	if last_primary_id is not None and last_primary_id in tracked:
		last_dist = np.linalg.norm(
			np.array(_centroid(tracked[last_primary_id])) - np.array(sensorPoint))
		candidate_dist = np.linalg.norm(
			np.array(_centroid(tracked[candidate])) - np.array(sensorPoint))
		if candidate != last_primary_id and candidate_dist < last_dist * (1.0 - PRIMARY_SWITCH_MARGIN):
			_last_primary_id = candidate
			return candidate
		_last_primary_id = last_primary_id
		return last_primary_id

	_last_primary_id = candidate
	return candidate

def evaluate_person(maskLabel, objectTempC, hasTemperature, feverThresholdC):
	if maskLabel == "Unknown":
		return "yellow", "ID WAITING: UNCERTAIN MASK"
	hasMask = maskLabel == "Mask"
	if hasMask and hasTemperature and objectTempC < feverThresholdC:
		return "green", "ID PASS: MASK + TEMP OK"
	if hasMask and not hasTemperature:
		return "yellow", "ID WAITING: TEMP UNKNOWN"
	if objectTempC is not None and objectTempC >= feverThresholdC:
		return "red", "ID FAIL: HIGH TEMP"
	return "red", "ID FAIL: NO MASK"

def load_detector_models():
	base_dir = os.path.dirname(os.path.abspath(__file__))
	prototxtPath = os.path.join(base_dir, "face_detector", "deploy.prototxt")
	weightsPath = os.path.join(
		base_dir, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
	maskPath = os.path.join(base_dir, "mask_detector.h5")

	missing = []
	if not os.path.isfile(maskPath):
		missing.append("mask_detector.h5")
	if not os.path.isfile(prototxtPath):
		missing.append("face_detector/deploy.prototxt")
	if not os.path.isfile(weightsPath):
		missing.append("face_detector/res10_300x300_ssd_iter_140000.caffemodel")

	if "mask_detector.h5" in missing:
		# #region agent log
		debug_log("detect_mask_video.py:load_detector_models", "mask model missing",
			{"missing": missing}, hypothesis_id="A")
		# #endregion
		raise FileNotFoundError(
			"mask_detector.h5 not found. Train or copy the model into the project root.")

	faceNet = None
	faceCascade = cv2.CascadeClassifier(
		os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
	if faceCascade.empty():
		raise RuntimeError("Failed to load Haar cascade face detector.")

	if os.path.isfile(prototxtPath) and os.path.isfile(weightsPath):
		try:
			faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
			# #region agent log
			debug_log("detect_mask_video.py:load_detector_models", "DNN face model loaded",
				{}, hypothesis_id="A")
			# #endregion
		except cv2.error as exc:
			debug_log("detect_mask_video.py:load_detector_models", "DNN load failed, using Haar",
				{"error": str(exc)}, hypothesis_id="A")
			faceNet = None
	else:
		debug_log("detect_mask_video.py:load_detector_models", "Caffe weights missing, using Haar",
			{"missing": [item for item in missing if "caffemodel" in item]},
			hypothesis_id="A")

	try:
		maskNet = load_model(maskPath)
	except Exception as exc:
		debug_log("detect_mask_video.py:load_detector_models", "mask model load failed",
			{"error": str(exc)}, hypothesis_id="A")
		raise RuntimeError(f"Failed to load mask_detector.h5: {exc}") from exc

	debug_log("detect_mask_video.py:load_detector_models", "models ready",
		{"dnn_enabled": faceNet is not None, "haar_enabled": not faceCascade.empty()},
		hypothesis_id="A")
	return faceNet, faceCascade, maskNet

def annotate_frame(frame, faceNet, maskNet, faceCascade, serialReader,
		tracker, feverThresholdC=37.5, mask_confidence_threshold=MASK_CONFIDENCE_THRESHOLD,
		log_cooldown_seconds=8):
	global _last_primary_id, _frame_times

	frame_start = time.time()
	reading = serialReader.get_reading()
	objectTempC = reading["object_temp_c"]
	ambientTempC = reading["ambient_temp_c"]
	serialStatus = reading["status"]
	(tempText, tempStatus, isHighTemp) = format_temperature_result(
		objectTempC, serialStatus, feverThresholdC)

	if hasattr(tracker, "set_frame_width"):
		tracker.set_frame_width(frame.shape[1])

	locs = detect_faces(frame, faceNet, faceCascade)
	tracked = tracker.update(locs)
	validLocs, preds = predict_masks(frame, list(tracked.values()), maskNet)

	predByBox = {}
	for box, pred in zip(validLocs, preds):
		predByBox[box] = pred

	cleanFrame = frame.copy()
	primaryID = choose_primary_person(tracked, frame.shape, last_primary_id=_last_primary_id)
	persons = []
	overallSignal = "yellow"
	status = "No face detected"
	statusColor = (0, 255, 255)
	skipped_no_pred = 0
	logged_violations = 0

	for objectID, box in tracked.items():
		pred = predByBox.get(box)
		if pred is None:
			skipped_no_pred += 1
			continue

		(startX, startY, endX, endY) = box
		maskLabel, maskConfidence = classify_mask(pred, mask_confidence_threshold)
		isPrimary = objectID == primaryID
		personTempC = objectTempC if isPrimary else None
		hasTemperature = personTempC is not None
		personTempText = tempText if isPrimary else "Temp: sensor mapped to primary ROI"
		personSignal, personStatus = evaluate_person(
			maskLabel, personTempC, hasTemperature, feverThresholdC)

		color = (0, 255, 0) if personSignal == "green" else (
			(0, 255, 255) if personSignal == "yellow" else (0, 0, 255))

		label = f"ID {objectID} | {maskLabel}: {maskConfidence:.1f}%"
		if isPrimary and personTempC is not None:
			label += f" | {personTempC:.1f} C"

		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

		persons.append({
			"id": int(objectID),
			"mask_label": maskLabel,
			"mask_confidence": maskConfidence,
			"temperature_c": float(personTempC) if personTempC is not None else None,
			"temp_text": personTempText,
			"signal": personSignal,
			"status_text": personStatus,
			"is_primary": isPrimary,
			"box": [int(startX), int(startY), int(endX), int(endY)],
		})

		if is_violation(maskLabel, personTempC, feverThresholdC):
			if should_log(int(objectID), cooldown_seconds=log_cooldown_seconds):
				personTempStatus = None
				if personTempC is not None:
					personTempStatus = "Fever" if personTempC >= feverThresholdC else "Normal"
				snapshotPath = save_violation_snapshot(
					cleanFrame, (startX, startY, endX, endY), maskLabel,
					personTempStatus, int(objectID))
				log_entry(maskLabel, personTempC, int(objectID),
					snapshotPath, feverThresholdC)
				logged_violations += 1
				# #region agent log
				debug_log("detect_mask_video.py:annotate_frame", "violation logged",
					{"person_id": int(objectID), "mask_label": maskLabel,
					 "temperature_c": personTempC, "snapshot": snapshotPath},
					hypothesis_id="C")
				# #endregion

		if isPrimary:
			overallSignal = personSignal
			status = personStatus.replace("ID ", "")
			statusColor = color

	if persons and primaryID is not None:
		primary = next(item for item in persons if item["is_primary"])
		if primary["signal"] == "green":
			status = f"GREEN SIGNAL: MASK + {tempStatus}"
			statusColor = (0, 255, 0)
			overallSignal = "green"
		elif primary["signal"] == "yellow":
			if primary["mask_label"] == "Unknown":
				status = "WAITING: UNCERTAIN MASK CLASSIFICATION"
			else:
				status = "WAITING: MASK DETECTED + TEMP UNKNOWN"
			statusColor = (0, 255, 255)
			overallSignal = "yellow"
		else:
			status = "RED SIGNAL: "
			status += "HIGH TEMP" if primary["temperature_c"] is not None and primary["temperature_c"] >= feverThresholdC else "NO MASK"
			statusColor = (0, 0, 255)
			overallSignal = "red"

	_frame_times.append(time.time() - frame_start)
	if len(_frame_times) > 30:
		_frame_times.pop(0)
	fps = round(1.0 / (sum(_frame_times) / len(_frame_times)), 1) if _frame_times else 0.0

	cv2.circle(frame, (25, 25), 12, statusColor, -1)
	cv2.putText(frame, status, (45, 32),
		cv2.FONT_HERSHEY_SIMPLEX, 0.55, statusColor, 2)
	cv2.putText(frame, tempText, (45, 56),
		cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
	cv2.putText(frame, serialStatus, (45, 80),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
	cv2.putText(frame, f"Tracked: {len(persons)} | FPS: {fps:.1f}", (45, 104),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

	primary = next((item for item in persons if item["is_primary"]), None)
	stats = {
		"mask_label": primary["mask_label"] if primary else "None",
		"mask_confidence": primary["mask_confidence"] if primary else None,
		"temperature_c": float(objectTempC) if objectTempC is not None else None,
		"ambient_c": float(ambientTempC) if ambientTempC is not None else None,
		"temp_status": tempStatus,
		"is_high_temp": bool(isHighTemp),
		"signal": overallSignal,
		"status_text": status,
		"serial_status": serialStatus,
		"temp_text": tempText,
		"fever_threshold_c": float(feverThresholdC),
		"tracked_count": len(persons),
		"primary_id": int(primaryID) if primaryID is not None else None,
		"persons": persons,
		"fps": fps,
		"debug_enabled": is_debug_enabled(),
	}

	# #region agent log
	debug_log("detect_mask_video.py:annotate_frame", "frame processed",
		{"faces_detected": len(locs), "tracked": len(tracked), "persons": len(persons),
		 "primary_id": primaryID, "skipped_no_pred": skipped_no_pred,
		 "logged_violations": logged_violations, "fps": fps, "signal": overallSignal},
		hypothesis_id="E")
	# #endregion

	return frame, stats

def main():
	args = parse_args()
	config = load_config()
	tempPort = args["temp_port"] or config.get("temp_port")
	tempBaud = args["temp_baud"] or config.get("temp_baud", 115200)
	feverThresholdC = config.get("fever_threshold_c", 37.5)
	serialEnabled = bool(config.get("serial_enabled", False))
	serialMode = config.get("serial_mode", "monitor")

	init_db()

	serialReader = SerialTemperatureReader(
		tempPort, tempBaud, enabled=serialEnabled, mode=serialMode)
	serialReader.start()
	tracker = CentroidTracker()

	faceNet, faceCascade, maskNet = load_detector_models()

	print("[INFO] starting video stream...", flush=True)
	vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	time.sleep(2.0)

	if not vs.isOpened():
		serialReader.stop()
		raise RuntimeError("Could not open webcam. Make sure no other app is using it.")

	consecutive_failures = 0
	while True:
		(grabbed, frame) = vs.read()
		if not grabbed or frame is None:
			consecutive_failures += 1
			print(f"[WARN] could not read frame from webcam ({consecutive_failures})", flush=True)
			if consecutive_failures >= 30:
				print("[ERROR] too many consecutive frame failures, stopping.", flush=True)
				break
			time.sleep(0.05)
			continue

		consecutive_failures = 0
		frame = imutils.resize(frame, width=640)
		try:
			frame, _ = annotate_frame(
				frame, faceNet, maskNet, faceCascade, serialReader,
				tracker, feverThresholdC,
				mask_confidence_threshold=config.get("mask_confidence_threshold", MASK_CONFIDENCE_THRESHOLD),
				log_cooldown_seconds=config.get("log_cooldown_seconds", 8))
		except Exception as exc:
			print(f"[WARN] frame processing error: {exc}", flush=True)
			continue

		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break

	cv2.destroyAllWindows()
	vs.release()
	serialReader.stop()

if __name__ == "__main__":
	main()
