import threading
import time

try:
	import serial
	from serial.tools import list_ports
except ImportError:
	serial = None
	list_ports = None


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
	def __init__(self, port=None, baud_rate=115200):
		self.port = port
		self.baud_rate = baud_rate
		self._lock = threading.Lock()
		self._stop_event = threading.Event()
		self._thread = None
		self._serial = None
		self.object_temp_c = None
		self.ambient_temp_c = None
		self.status = "Serial: not started"
		self.last_update_at = None

	def start(self):
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

	def reconnect(self, port=None, baud_rate=None):
		if port is not None:
			self.port = port
		if baud_rate is not None:
			self.baud_rate = baud_rate

		self._close_serial()
		with self._lock:
			self.object_temp_c = None
			self.ambient_temp_c = None

	def get_reading(self):
		with self._lock:
			return {
				"object_temp_c": self.object_temp_c,
				"ambient_temp_c": self.ambient_temp_c,
				"status": self.status,
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
			time.sleep(2.0)
			self._serial.reset_input_buffer()
			with self._lock:
				self.status = f"Serial: connected on {port}"
			return True
		except serial.SerialException as exc:
			with self._lock:
				if "Access is denied" in str(exc):
					self.status = f"Serial: {port} blocked - close Arduino Serial Monitor"
				else:
					self.status = f"Serial: could not open {port}"
			self._close_serial()
			return False

	def _read_loop(self):
		while not self._stop_event.is_set():
			if self._serial is None or not getattr(self._serial, "is_open", False):
				if not self._open_serial():
					time.sleep(2.0)
					continue

			try:
				while self._serial.in_waiting:
					line = self._serial.readline().decode("utf-8", errors="ignore").strip()
					parsed = parse_temperature_line(line)
					if parsed is not None:
						objectTempC, ambientTempC = parsed
						with self._lock:
							self.object_temp_c = objectTempC
							self.ambient_temp_c = ambientTempC
							self.last_update_at = time.time()
			except serial.SerialException:
				self._close_serial()
				with self._lock:
					self.status = "Serial: read error, reconnecting"
				time.sleep(1.0)
				continue

			time.sleep(0.05)

		self._close_serial()
