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
from serial_reader import SerialTemperatureReader, find_temperature_port
from db import init_db, log_entry, should_log
from snapshot import save_violation_snapshot

FACE_CONFIDENCE = 0.2

def parse_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("--temp-port", type=str, default=None,
		help="ESP32 serial port, for example COM3. If omitted, config.json is used.")
	ap.add_argument("--temp-baud", type=int, default=None,
		help="ESP32 serial baud rate.")
	return vars(ap.parse_args())

def format_temperature_result(objectTempC, serialStatus, feverThresholdC):
	if objectTempC is None:
		if "blocked" in serialStatus.lower():
			return ("Temp: close Arduino Serial Monitor", "TEMP UNKNOWN", False)
		if "connected" in serialStatus.lower():
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

def detect_faces(frame, faceNet, faceCascade):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()

	locs = []
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > FACE_CONFIDENCE:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			if endX > startX and endY > startY:
				locs.append((startX, startY, endX, endY))

	if len(locs) == 0:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		fallbackFaces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=1.05,
			minNeighbors=4,
			minSize=(45, 45)
		)
		for (x, y, fw, fh) in fallbackFaces:
			locs.append((x, y, x + fw, y + fh))

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
	preds = maskNet.predict(faces, batch_size=32, verbose=0)
	return validLocs, preds

def _centroid(rect):
	(startX, startY, endX, endY) = rect
	return (int((startX + endX) / 2.0), int((startY + endY) / 2.0))

def choose_primary_person(tracked, frameShape, sensorPoint=None):
	if not tracked:
		return None

	(h, w) = frameShape[:2]
	if sensorPoint is None:
		sensorPoint = (w // 2, h // 3)

	return min(
		tracked.keys(),
		key=lambda objectID: np.linalg.norm(
			np.array(_centroid(tracked[objectID])) - np.array(sensorPoint)
		),
	)

def evaluate_person(maskLabel, objectTempC, hasTemperature, feverThresholdC):
	hasMask = maskLabel == "Mask"
	if hasMask and hasTemperature and objectTempC < feverThresholdC:
		return "green", f"ID PASS: MASK + TEMP OK"
	if hasMask and not hasTemperature:
		return "yellow", "ID WAITING: TEMP UNKNOWN"
	if objectTempC is not None and objectTempC >= feverThresholdC:
		return "red", "ID FAIL: HIGH TEMP"
	return "red", "ID FAIL: NO MASK"

def load_detector_models():
	prototxtPath = r"face_detector\deploy.prototxt"
	weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
	faceCascade = cv2.CascadeClassifier(
		os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
	)
	maskNet = load_model("mask_detector.h5")
	return faceNet, faceCascade, maskNet

def annotate_frame(frame, faceNet, maskNet, faceCascade, serialReader,
		tracker, feverThresholdC=37.5):
	reading = serialReader.get_reading()
	objectTempC = reading["object_temp_c"]
	ambientTempC = reading["ambient_temp_c"]
	serialStatus = reading["status"]
	(tempText, tempStatus, isHighTemp) = format_temperature_result(
		objectTempC, serialStatus, feverThresholdC)

	locs = detect_faces(frame, faceNet, faceCascade)
	tracked = tracker.update(locs)
	validLocs, preds = predict_masks(frame, list(tracked.values()), maskNet)

	predByBox = {}
	for box, pred in zip(validLocs, preds):
		predByBox[box] = pred

	cleanFrame = frame.copy()
	primaryID = choose_primary_person(tracked, frame.shape)
	persons = []
	overallSignal = "yellow"
	status = "No face detected"
	statusColor = (0, 255, 255)

	for objectID, box in tracked.items():
		pred = predByBox.get(box)
		if pred is None:
			continue

		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		maskLabel = "Mask" if mask > withoutMask else "No Mask"
		maskConfidence = float(max(mask, withoutMask) * 100)
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

		if should_log(int(objectID)):
			personTempStatus = None
			if personTempC is not None:
				personTempStatus = "Fever" if personTempC >= feverThresholdC else "Normal"
			snapshotPath = save_violation_snapshot(
				cleanFrame, (startX, startY, endX, endY), maskLabel,
				personTempStatus, int(objectID))
			log_entry(maskLabel, personTempC, int(objectID),
				snapshotPath, feverThresholdC)

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
			status = "WAITING: MASK DETECTED + TEMP UNKNOWN"
			statusColor = (0, 255, 255)
			overallSignal = "yellow"
		else:
			status = "RED SIGNAL: "
			status += "HIGH TEMP" if primary["temperature_c"] is not None and primary["temperature_c"] >= feverThresholdC else "NO MASK"
			statusColor = (0, 0, 255)
			overallSignal = "red"

	cv2.circle(frame, (25, 25), 12, statusColor, -1)
	cv2.putText(frame, status, (45, 32),
		cv2.FONT_HERSHEY_SIMPLEX, 0.55, statusColor, 2)
	cv2.putText(frame, tempText, (45, 56),
		cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
	cv2.putText(frame, serialStatus, (45, 80),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
	cv2.putText(frame, f"Tracked: {len(persons)}", (45, 104),
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
	}

	return frame, stats

def main():
	args = parse_args()
	config = load_config()
	tempPort = args["temp_port"] or config.get("temp_port")
	tempBaud = args["temp_baud"] or config.get("temp_baud", 115200)
	feverThresholdC = config.get("fever_threshold_c", 37.5)

	init_db()

	serialReader = SerialTemperatureReader(tempPort, tempBaud)
	serialReader.start()
	tracker = CentroidTracker()

	faceNet, faceCascade, maskNet = load_detector_models()

	print("[INFO] starting video stream...", flush=True)
	vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	time.sleep(2.0)

	if not vs.isOpened():
		serialReader.stop()
		raise RuntimeError("Could not open webcam. Make sure no other app is using it.")

	while True:
		(grabbed, frame) = vs.read()
		if not grabbed or frame is None:
			print("[WARN] could not read frame from webcam")
			break

		frame = imutils.resize(frame, width=640)
		frame, _ = annotate_frame(
			frame, faceNet, maskNet, faceCascade, serialReader,
			tracker, feverThresholdC)

		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break

	cv2.destroyAllWindows()
	vs.release()
	serialReader.stop()

if __name__ == "__main__":
	main()
