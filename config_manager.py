import json
import os
import threading

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

DEFAULT_CONFIG = {
	"fever_threshold_c": 37.5,
	"temp_port": "COM4",
	"temp_baud": 115200,
	"serial_enabled": False,
	"serial_mode": "monitor",
	"debug": False,
	"mask_confidence_threshold": 70.0,
	"log_cooldown_seconds": 8,
}

_config_lock = threading.Lock()


def load_config():
	with _config_lock:
		if not os.path.exists(CONFIG_PATH):
			save_config(DEFAULT_CONFIG)
			return DEFAULT_CONFIG.copy()

		with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
			data = json.load(handle)

		merged = DEFAULT_CONFIG.copy()
		merged.update(data)
		return merged


def save_config(config):
	with _config_lock:
		with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
			json.dump(config, handle, indent=2)
		return config.copy()


def update_config(updates):
	config = load_config()
	config.update(updates)
	return save_config(config)
