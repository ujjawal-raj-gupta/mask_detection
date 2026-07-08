import re
import threading
import time
from debug_utils import debug_log

try:
	import serial
	from serial.tools import list_ports
except ImportError:
	serial = None
	list_ports = None


def parse_status_line(line):
	if line.startswith("BOOT:"):
		return "ESP32 sketch running"
	if line.startswith("STATUS:"):
		return line.replace("STATUS:", "").strip()
	if line.startswith("READY:"):
		return line.replace("READY:", "Sensor ready:").strip()
	if line.startswith("ERROR:"):
		return line.replace("ERROR:", "Sensor error:").strip()
	return None


def parse_temperature_line(line):
	if not line.startswith("TEMP_C:"):
		return None

	values = {}
	for part in line.split(","):
		key, _, value = part.partition(":")
		if key and value:
			try:
				values[key] = float(value)
			except ValueError:
				return None

	if "TEMP_C" not in values:
		return None

	return (values["TEMP_C"], values.get("AMBIENT_C"))


def parse_serial_monitor_line(line):
	"""Parse a line copied from Arduino Serial Monitor (MLX, DHT, or plain)."""
	line = line.strip()
	if not line:
		return None

	parsed = parse_temperature_line(line)
	if parsed is not None:
		return parsed

	lower = line.lower()
	patterns = [
		r"temp(?:erature)?\s*[:=]\s*(-?\d+(?:\.\d+)?)",
		r"(-?\d+(?:\.\d+)?)\s*(?:°|deg)?\s*c",
	]
	for pattern in patterns:
		match = re.search(pattern, lower)
		if match:
			try:
				return (float(match.group(1)), None)
			except ValueError:
				continue

	if re.fullmatch(r"-?\d+(?:\.\d+)?", line):
		try:
			return (float(line), None)
		except ValueError:
			return None

	return None


def apply_reading(reader, object_temp_c, ambient_temp_c=None, status=None):
	with reader._lock:
		reader.object_temp_c = object_temp_c
		reader.ambient_temp_c = ambient_temp_c
		reader.last_update_at = time.time()
		if status:
			reader.status = status


def find_temperature_port(preferred_port=None):
	if preferred_port:
		return preferred_port

	if list_ports is None:
		return None

	keywords = ("usb", "uart", "cp210", "ch340", "silicon", "esp32")
	for port in list_ports.comports():
		description = f"{port.description} {port.manufacturer} {port.hwid}".lower()
		if any(keyword in description for keyword in keywords):
			return port.device

	return None


def list_serial_ports():
	if list_ports is None:
		return []

	return [
		{
			"device": port.device,
			"description": port.description or port.device,
		}
		for port in list_ports.comports()
	]


class SerialTemperatureReader:
	def __init__(self, port=None, baud_rate=115200, enabled=True, mode="direct"):
		self.port = port
		self.baud_rate = baud_rate
		self.enabled = enabled
		self.mode = mode if mode in ("direct", "monitor") else "direct"
		self._lock = threading.Lock()
		self._stop_event = threading.Event()
		self._thread = None
		self._serial = None
		self.object_temp_c = None
		self.ambient_temp_c = None
		if not enabled:
			self.status = "Serial: on hold"
		elif self.mode == "monitor":
			self.status = "Serial: Serial Monitor mode — paste readings below"
		else:
			self.status = "Serial: not started"
		self.last_update_at = None
		self.stale_after_seconds = 30.0

	def feed_line(self, line):
		"""Accept text pasted from Arduino Serial Monitor."""
		for part in line.replace("\r", "\n").split("\n"):
			part = part.strip()
			if not part:
				continue
			parsed = parse_serial_monitor_line(part)
			if parsed is None:
				status_msg = parse_status_line(part)
				if status_msg:
					with self._lock:
						self.status = f"Serial: {status_msg}"
				continue

			objectTempC, ambientTempC = parsed
			apply_reading(
				self, objectTempC, ambientTempC,
				status="Serial: live from Serial Monitor paste")
			return True
		return False

	def start(self):
		if not self.enabled:
			with self._lock:
				self.status = "Serial: on hold"
			return

		if self.mode == "monitor":
			with self._lock:
				if self.object_temp_c is None:
					self.status = "Serial: Serial Monitor mode — paste readings below"
				else:
					self.status = "Serial: live from Serial Monitor paste"
			return

		if self._thread and self._thread.is_alive():
			return

		self._stop_event.clear()
		self._thread = threading.Thread(target=self._read_loop, daemon=True)
		self._thread.start()

	def stop(self):
		self._stop_event.set()
		if self._thread:
			self._thread.join(timeout=2.0)
		self._close_serial()

	def reconnect(self, port=None, baud_rate=None, enabled=None, mode=None):
		if enabled is not None:
			self.enabled = enabled
		if port is not None:
			self.port = port
		if baud_rate is not None:
			self.baud_rate = baud_rate
		if mode is not None:
			self.mode = mode if mode in ("direct", "monitor") else self.mode

		self._stop_event.set()
		if self._thread and self._thread.is_alive():
			self._thread.join(timeout=2.0)
		self._thread = None
		self._stop_event.clear()
		self._close_serial()
		with self._lock:
			if not self.enabled:
				self.status = "Serial: on hold"
			elif self.mode == "monitor":
				self.status = (
					"Serial: live from Serial Monitor paste"
					if self.object_temp_c is not None
					else "Serial: Serial Monitor mode — paste readings below"
				)
			else:
				self.object_temp_c = None
				self.ambient_temp_c = None
				self.status = "Serial: not started"
		if self.enabled:
			self.start()

	def get_reading(self):
		with self._lock:
			status = self.status
			if (self.enabled and self.last_update_at is not None
					and time.time() - self.last_update_at > self.stale_after_seconds):
				status = "Serial: connected, waiting for sensor data"
			return {
				"object_temp_c": self.object_temp_c,
				"ambient_temp_c": self.ambient_temp_c,
				"status": status,
				"last_update_at": self.last_update_at,
			}

	def _close_serial(self):
		if self._serial is not None:
			try:
				self._serial.close()
			except serial.SerialException:
				pass
			self._serial = None

	def _open_serial(self):
		if serial is None:
			with self._lock:
				self.status = "Serial: pyserial not installed"
			return False

		port = find_temperature_port(self.port)
		if port is None:
			with self._lock:
				self.status = "Serial: ESP32 port not found"
			return False

		try:
			self._serial = serial.Serial(port, self.baud_rate, timeout=1)
			time.sleep(2.5)
			with self._lock:
				self.status = f"Serial: connected on {port}, starting ESP32..."

			deadline = time.time() + 4.0
			while time.time() < deadline:
				line = self._serial.readline().decode("utf-8", errors="ignore").strip()
				if not line or line.startswith("load:") or "SPI" in line or "entry" in line:
					continue
				parsed = parse_temperature_line(line)
				if parsed is not None:
					objectTempC, ambientTempC = parsed
					with self._lock:
						self.object_temp_c = objectTempC
						self.ambient_temp_c = ambientTempC
						self.last_update_at = time.time()
						self.status = f"Serial: live on {port}"
					return True
				status_msg = parse_status_line(line)
				if status_msg:
					with self._lock:
						self.status = f"Serial: {status_msg}"
			with self._lock:
				if self.object_temp_c is None:
					self.status = (
						f"Serial: connected on {port} — upload "
						"esp32_mlx90614_temperature.ino to ESP32"
					)
			return True
		except serial.SerialException as exc:
			with self._lock:
				if "Access is denied" in str(exc):
					self.status = f"Serial: {port} blocked - close Arduino Serial Monitor"
				else:
					self.status = f"Serial: could not open {port}"
			# #region agent log
			debug_log("serial_reader.py:_open_serial", "serial open failed",
				{"port": port, "error": str(exc)}, hypothesis_id="G")
			# #endregion
			self._close_serial()
			return False

	def _read_loop(self):
		while not self._stop_event.is_set():
			if not self.enabled:
				with self._lock:
					self.status = "Serial: on hold"
				time.sleep(1.0)
				continue

			if self.mode == "monitor":
				time.sleep(0.5)
				continue

			if self._serial is None or not getattr(self._serial, "is_open", False):
				if not self._open_serial():
					time.sleep(2.0)
					continue

			try:
				line = self._serial.readline().decode("utf-8", errors="ignore").strip()
				if not line:
					time.sleep(0.05)
					continue

				parsed = parse_temperature_line(line)
				if parsed is not None:
					objectTempC, ambientTempC = parsed
					with self._lock:
						self.object_temp_c = objectTempC
						self.ambient_temp_c = ambientTempC
						self.last_update_at = time.time()
						port = find_temperature_port(self.port) or self.port
						self.status = f"Serial: live on {port}"
					continue

				status_msg = parse_status_line(line)
				if status_msg:
					with self._lock:
						self.status = f"Serial: {status_msg}"
			except serial.SerialException:
				self._close_serial()
				with self._lock:
					self.status = "Serial: read error, reconnecting"
				time.sleep(1.0)
				continue

			time.sleep(0.05)

		self._close_serial()
