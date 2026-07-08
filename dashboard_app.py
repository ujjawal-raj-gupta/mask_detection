import argparse
import atexit
import os
import socket
import threading
import time

import cv2
import imutils
import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template, request, send_file, send_from_directory, url_for
from flask_socketio import SocketIO

from centroid_tracker import CentroidTracker
from config_manager import load_config
from db import CSV_PATH, delete_all_entries, delete_entry, fetch_recent, fetch_today_stats, init_db
from debug_utils import debug_log, is_debug_enabled
from detect_mask_video import annotate_frame, load_detector_models
from serial_reader import SerialTemperatureReader
from snapshot import VIOLATIONS_DIR, VIOLATIONS_DIRNAME

app = Flask(__name__)
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
	"serial_status": "Serial: on hold",
	"temp_text": "Temp: waiting",
	"fever_threshold_c": 37.5,
	"tracked_count": 0,
	"primary_id": None,
	"persons": [],
	"camera_ready": False,
}

detector_running = False
serial_reader = None
_last_serial_cfg = None
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
		"serial_enabled": bool(config.get("serial_enabled", False)),
		"serial_mode": config.get("serial_mode", "monitor"),
		"mask_confidence_threshold": float(config.get("mask_confidence_threshold", 70.0)),
		"log_cooldown_seconds": int(config.get("log_cooldown_seconds", 8)),
		"debug": bool(config.get("debug", False)),
	}


def ensure_serial_reader():
	global serial_reader, _last_serial_cfg
	config = get_runtime_config()
	cfg_key = (
		config["serial_enabled"], config["temp_port"],
		config["temp_baud"], config["serial_mode"],
	)

	if serial_reader is not None and _last_serial_cfg == cfg_key:
		return

	_last_serial_cfg = cfg_key

	if serial_reader is None:
		serial_reader = SerialTemperatureReader(
			config["temp_port"], config["temp_baud"],
			enabled=config["serial_enabled"],
			mode=config["serial_mode"])
		serial_reader.start()
	else:
		serial_reader.reconnect(
			config["temp_port"], config["temp_baud"],
			enabled=config["serial_enabled"],
			mode=config["serial_mode"])


def detection_loop():
	global latest_frame, latest_jpeg, latest_stats, detector_running, tracker
	global faceNet, faceCascade, maskNet, serial_reader

	def push_status(status_text, serial_status=None):
		with lock:
			latest_stats["status_text"] = status_text
			if serial_status is not None:
				latest_stats["serial_status"] = serial_status
		socketio.emit("telemetry_update", serialize_stats(latest_stats))

	push_status("Starting detector...", "Serial: on hold")
	ensure_serial_reader()
	reading = serial_reader.get_reading()
	if get_runtime_config()["serial_enabled"]:
		push_status("Connecting ESP32 serial...", reading["status"])
		reading = serial_reader.get_reading()
	push_status("Loading AI models...", reading["status"])

	if faceNet is None or faceCascade is None or maskNet is None:
		try:
			loadedFaceNet, loadedCascade, loadedMaskNet = load_detector_models()
			faceNet, faceCascade, maskNet = loadedFaceNet, loadedCascade, loadedMaskNet
		except Exception as model_exc:
			with lock:
				latest_stats = {
					**latest_stats,
					"status_text": f"Model load failed: {model_exc}",
					"signal": "red",
				}
			socketio.emit("telemetry_update", serialize_stats(latest_stats))
			debug_log("dashboard_app.py:detection_loop", "model load failed",
				{"error": str(model_exc)}, hypothesis_id="A")
			return

	tracker = CentroidTracker()
	push_status("Opening webcam...", serial_reader.get_reading()["status"])

	vs = None
	vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	for attempt in range(10):
		if vs.isOpened() and vs.read()[0]:
			# #region agent log
			debug_log("dashboard_app.py:detection_loop", "camera opened",
				{"attempt": attempt + 1}, hypothesis_id="D")
			# #endregion
			break
		time.sleep(0.2)

	if vs is None or not vs.isOpened():
		with lock:
			latest_stats = {
				**latest_stats,
				"status_text": "Webcam not available",
				"signal": "red",
			}
		detector_running = False
		socketio.emit("telemetry_update", serialize_stats(latest_stats))
		debug_log("dashboard_app.py:detection_loop", "camera unavailable",
			{}, hypothesis_id="D")
		return

	detector_running = True
	print("[INFO] dashboard detection loop started", flush=True)
	consecutive_failures = 0

	try:
		while detector_running:
			config = get_runtime_config()
			ensure_serial_reader()
			try:
				(grabbed, frame) = vs.read()
				if not grabbed or frame is None:
					consecutive_failures += 1
					if consecutive_failures % 20 == 0:
						debug_log("dashboard_app.py:detection_loop", "empty frame",
							{"consecutive_failures": consecutive_failures},
							hypothesis_id="D")
					time.sleep(0.05)
					continue

				consecutive_failures = 0
				frame = imutils.resize(frame, width=480)
				frame, stats = annotate_frame(
					frame, faceNet, maskNet, faceCascade, serial_reader,
					tracker, config["fever_threshold_c"],
					mask_confidence_threshold=config["mask_confidence_threshold"],
					log_cooldown_seconds=config["log_cooldown_seconds"])
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
		if vs is not None:
			vs.release()
		detector_running = False
		if serial_reader is not None:
			serial_reader.stop()
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


@app.route("/logs")
def logs():
	entries = fetch_recent(50)
	stats = fetch_today_stats()
	message = request.args.get("message")
	error = request.args.get("error")
	return render_template(
		"logs.html",
		entries=entries,
		stats=stats,
		message=message,
		error=error,
	)


@app.route("/logs/delete/<int:entry_id>", methods=["POST"])
def logs_delete_entry(entry_id):
	if delete_entry(entry_id):
		return redirect(url_for("logs", message=f"Deleted entry #{entry_id}."))
	return redirect(url_for("logs", error=f"Entry #{entry_id} not found."))


@app.route("/logs/delete-all", methods=["POST"])
def logs_delete_all():
	deleted = delete_all_entries()
	return redirect(url_for("logs", message=f"Deleted {deleted} log entries."))


@app.route("/logs/export.csv")
def logs_export_csv():
	init_db()
	if not os.path.isfile(CSV_PATH):
		return Response("No screening log CSV found yet.", status=404, mimetype="text/plain")
	return send_file(
		CSV_PATH,
		mimetype="text/csv",
		as_attachment=True,
		download_name="screening_log.csv",
	)


@app.route(f"/{VIOLATIONS_DIRNAME}/<path:filename>")
def violations(filename):
	return send_from_directory(VIOLATIONS_DIR, filename)


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


@app.route("/api/serial-input", methods=["POST"])
def api_serial_input():
	global serial_reader
	payload = request.get_json(silent=True) or {}
	line = payload.get("line") or payload.get("text") or request.get_data(as_text=True)
	line = (line or "").strip()

	if not line:
		return jsonify({"ok": False, "error": "No serial line provided"}), 400

	ensure_serial_reader()
	if serial_reader is None:
		return jsonify({"ok": False, "error": "Serial reader not initialized"}), 500

	if not serial_reader.feed_line(line):
		return jsonify({
			"ok": False,
			"error": "Could not parse temperature from line",
			"hint": "Paste e.g. TEMP_C:36.5,AMBIENT_C:28.0 or Temperature: 36.5",
		}), 400

	reading = serial_reader.get_reading()
	with lock:
		socketio.emit("telemetry_update", serialize_stats(latest_stats))

	return jsonify({
		"ok": True,
		"object_temp_c": reading["object_temp_c"],
		"ambient_c": reading["ambient_temp_c"],
		"status": reading["status"],
	})


@socketio.on("connect")
def handle_connect():
	with lock:
		socketio.emit("telemetry_update", serialize_stats(latest_stats))


def parse_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("--host", type=str, default="127.0.0.1")
	ap.add_argument("--port", type=int, default=5000)
	return vars(ap.parse_args())


def ensure_single_instance():
	lock_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	try:
		lock_sock.bind(("127.0.0.1", 50505))
		lock_sock.listen(1)
	except OSError:
		print("[ERROR] Another dashboard instance is already running.")
		print("        Close it first so the webcam and COM port are not locked.")
		raise SystemExit(1)
	return lock_sock


def shutdown_serial():
	global serial_reader
	if serial_reader is not None:
		serial_reader.stop()


if __name__ == "__main__":
	args = parse_args()
	_instance_lock = ensure_single_instance()
	atexit.register(shutdown_serial)
	load_config()
	init_db()
	print("[INFO] preloading AI models...", flush=True)
	try:
		faceNet, faceCascade, maskNet = load_detector_models()
		print("[INFO] models loaded", flush=True)
	except Exception as model_exc:
		print(f"[ERROR] model preload failed: {model_exc}", flush=True)
		debug_log("dashboard_app.py:main", "startup model preload failed",
			{"error": str(model_exc)}, hypothesis_id="A")
		faceNet = faceCascade = maskNet = None

	thread = threading.Thread(target=detection_loop, daemon=True)
	thread.start()

	print(f"[INFO] open dashboard at http://{args['host']}:{args['port']}", flush=True)
	if is_debug_enabled():
		print("[INFO] debug mode enabled (config debug=true or SCREENING_DEBUG=1)", flush=True)
	socketio.run(app, host=args["host"], port=args["port"], debug=False, allow_unsafe_werkzeug=True)
