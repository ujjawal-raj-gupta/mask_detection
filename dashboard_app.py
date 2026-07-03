import argparse
import threading
import time

import cv2
import imutils
import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
from flask_socketio import SocketIO

from centroid_tracker import CentroidTracker
from config_manager import load_config, save_config, update_config
from detect_mask_video import annotate_frame, load_detector_models
from serial_reader import SerialTemperatureReader, list_serial_ports

app = Flask(__name__)
app.secret_key = "mask-detector-dashboard-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

lock = threading.Lock()
latest_frame = None
latest_jpeg = None
latest_stats = {
	"mask_label": "None",
	"mask_confidence": None,
	"temperature_c": None,
	"ambient_c": None,
	"temp_status": "TEMP UNKNOWN",
	"is_high_temp": False,
	"signal": "yellow",
	"status_text": "Starting...",
	"serial_status": "Serial: not connected",
	"temp_text": "Temp: waiting",
	"fever_threshold_c": 37.5,
	"tracked_count": 0,
	"primary_id": None,
	"persons": [],
	"camera_ready": False,
}

detector_running = False
serial_reader = None
tracker = None
faceNet = None
faceCascade = None
maskNet = None


def serialize_stats(stats):
	safe = {}
	for key, value in stats.items():
		if isinstance(value, (np.integer,)):
			safe[key] = int(value)
		elif isinstance(value, (np.floating,)):
			safe[key] = float(value)
		elif isinstance(value, (np.bool_,)):
			safe[key] = bool(value)
		elif isinstance(value, list):
			safe[key] = [serialize_stats(item) if isinstance(item, dict) else item for item in value]
		elif isinstance(value, dict):
			safe[key] = serialize_stats(value)
		else:
			safe[key] = value
	return safe


def get_runtime_config():
	config = load_config()
	return {
		"fever_threshold_c": float(config.get("fever_threshold_c", 37.5)),
		"temp_port": config.get("temp_port"),
		"temp_baud": int(config.get("temp_baud", 115200)),
	}


def ensure_serial_reader():
	global serial_reader
	config = get_runtime_config()

	if serial_reader is None:
		serial_reader = SerialTemperatureReader(
			config["temp_port"], config["temp_baud"])
		serial_reader.start()
	else:
		serial_reader.reconnect(config["temp_port"], config["temp_baud"])


def detection_loop():
	global latest_frame, latest_jpeg, latest_stats, detector_running, tracker
	global faceNet, faceCascade, maskNet, serial_reader

	def push_status(status_text, serial_status=None):
		with lock:
			latest_stats["status_text"] = status_text
			if serial_status is not None:
				latest_stats["serial_status"] = serial_status
		socketio.emit("telemetry_update", serialize_stats(latest_stats))

	push_status("Connecting ESP32 serial...")
	ensure_serial_reader()
	reading = serial_reader.get_reading()
	push_status("Loading AI models...", reading["status"])

	if faceNet is None or faceCascade is None or maskNet is None:
		loadedFaceNet, loadedCascade, loadedMaskNet = load_detector_models()
		faceNet, faceCascade, maskNet = loadedFaceNet, loadedCascade, loadedMaskNet

	tracker = CentroidTracker()
	push_status("Opening webcam...", serial_reader.get_reading()["status"])

	vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	for _ in range(10):
		if vs.isOpened() and vs.read()[0]:
			break
		time.sleep(0.2)

	if not vs.isOpened():
		with lock:
			latest_stats = {
				**latest_stats,
				"status_text": "Webcam not available",
				"signal": "red",
			}
		detector_running = False
		socketio.emit("telemetry_update", serialize_stats(latest_stats))
		return

	detector_running = True
	print("[INFO] dashboard detection loop started", flush=True)

	try:
		while detector_running:
			config = get_runtime_config()
			try:
				(grabbed, frame) = vs.read()
				if not grabbed or frame is None:
					time.sleep(0.05)
					continue

				frame = imutils.resize(frame, width=480)
				frame, stats = annotate_frame(
					frame, faceNet, maskNet, faceCascade, serial_reader,
					tracker, config["fever_threshold_c"])
				stats["camera_ready"] = True

				ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
				jpegBytes = buffer.tobytes() if ok else None

				with lock:
					latest_frame = frame.copy()
					if jpegBytes is not None:
						latest_jpeg = jpegBytes
					latest_stats = serialize_stats(stats)

				socketio.emit("telemetry_update", latest_stats)
			except Exception as frame_exc:
				print(f"[WARN] frame processing error: {frame_exc}", flush=True)
				with lock:
					latest_stats = {
						**latest_stats,
						"status_text": f"Frame error: {frame_exc}",
						"camera_ready": latest_stats.get("camera_ready", False),
					}
				socketio.emit("telemetry_update", serialize_stats(latest_stats))
				time.sleep(0.05)
	except Exception as exc:
		with lock:
			latest_stats = {
				**latest_stats,
				"status_text": f"Detector error: {exc}",
				"signal": "red",
			}
		socketio.emit("telemetry_update", serialize_stats(latest_stats))
		print(f"[ERROR] dashboard detection loop failed: {exc}", flush=True)
	finally:
		vs.release()
		detector_running = False
		print("[INFO] dashboard detection loop stopped", flush=True)


def generate_frames():
	while True:
		jpegBytes = None
		with lock:
			if latest_jpeg is not None:
				jpegBytes = latest_jpeg

		if jpegBytes is None:
			ok, buffer = cv2.imencode(".jpg", _placeholder_frame(), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
			if ok:
				jpegBytes = buffer.tobytes()

		if jpegBytes is None:
			time.sleep(0.05)
			continue

		yield (
			b"--frame\r\n"
			b"Content-Type: image/jpeg\r\n\r\n" + jpegBytes + b"\r\n"
		)
		time.sleep(0.05)


def _placeholder_frame():
	frame = np.zeros((360, 640, 3), dtype="uint8")
	return frame


def is_admin():
	return session.get("admin_authenticated") is True


@app.after_request
def add_no_cache_headers(response):
	if response.content_type and "text/html" in response.content_type:
		response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
		response.headers["Pragma"] = "no-cache"
		response.headers["Expires"] = "0"
	return response


@app.route("/")
def index():
	return render_template("dashboard.html")


@app.route("/settings", methods=["GET", "POST"])
def settings():
	config = load_config()

	if request.method == "POST":
		password = request.form.get("password", "")
		if not is_admin():
			if password == config.get("admin_password"):
				session["admin_authenticated"] = True
			else:
				return render_template(
					"settings.html",
					config=config,
					ports=list_serial_ports(),
					error="Invalid admin password.",
					authenticated=False,
				)

		if not is_admin():
			return redirect(url_for("settings"))

		newPassword = request.form.get("new_password", "").strip()
		updates = {
			"fever_threshold_c": float(request.form.get("fever_threshold_c", 37.5)),
			"temp_port": request.form.get("temp_port") or config.get("temp_port"),
			"temp_baud": int(request.form.get("temp_baud", config.get("temp_baud", 115200))),
		}
		if newPassword:
			updates["admin_password"] = newPassword

		updated = update_config(updates)
		if serial_reader is not None:
			serial_reader.reconnect(updated["temp_port"], updated["temp_baud"])

		return render_template(
			"settings.html",
			config=updated,
			ports=list_serial_ports(),
			message="Settings saved successfully.",
			authenticated=True,
		)

	return render_template(
		"settings.html",
		config=config,
		ports=list_serial_ports(),
		authenticated=is_admin(),
	)


@app.route("/settings/logout")
def settings_logout():
	session.pop("admin_authenticated", None)
	return redirect(url_for("settings"))


@app.route("/video_feed")
def video_feed():
	return Response(
		generate_frames(),
		mimetype="multipart/x-mixed-replace; boundary=frame",
		headers={
			"Cache-Control": "no-cache, no-store, must-revalidate",
			"Pragma": "no-cache",
			"Expires": "0",
		},
	)


@app.route("/api/frame.jpg")
def api_frame():
	with lock:
		jpegBytes = latest_jpeg

	if jpegBytes is None:
		ok, buffer = cv2.imencode(".jpg", _placeholder_frame(), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
		jpegBytes = buffer.tobytes() if ok else b""

	return Response(jpegBytes, mimetype="image/jpeg", headers={
		"Cache-Control": "no-cache, no-store, must-revalidate",
	})


@app.route("/api/status")
def api_status():
	with lock:
		return jsonify(serialize_stats(latest_stats))


@app.route("/api/config")
def api_config():
	if not is_admin():
		return jsonify({"error": "Unauthorized"}), 401
	return jsonify(load_config())


@socketio.on("connect")
def handle_connect():
	with lock:
		socketio.emit("telemetry_update", serialize_stats(latest_stats))


def parse_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("--host", type=str, default="127.0.0.1")
	ap.add_argument("--port", type=int, default=5000)
	return vars(ap.parse_args())


if __name__ == "__main__":
	args = parse_args()
	load_config()
	print("[INFO] preloading AI models...", flush=True)
	faceNet, faceCascade, maskNet = load_detector_models()
	print("[INFO] models loaded", flush=True)

	thread = threading.Thread(target=detection_loop, daemon=True)
	thread.start()

	print(f"[INFO] open dashboard at http://{args['host']}:{args['port']}", flush=True)
	print(f"[INFO] admin settings at http://{args['host']}:{args['port']}/settings", flush=True)
	socketio.run(app, host=args["host"], port=args["port"], debug=False, allow_unsafe_werkzeug=True)
