import json
import os
import time

SESSION_ID = "bd83dd"
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug-bd83dd.log")

_debug_enabled = None


def is_debug_enabled():
	global _debug_enabled
	if _debug_enabled is not None:
		return _debug_enabled

	env_val = os.environ.get("SCREENING_DEBUG", "").strip().lower()
	if env_val in ("1", "true", "yes", "on"):
		_debug_enabled = True
		return True

	try:
		from config_manager import load_config
		_debug_enabled = bool(load_config().get("debug", False))
	except Exception:
		_debug_enabled = False
	return _debug_enabled


def debug_log(location, message, data=None, hypothesis_id=None, run_id="pre-fix"):
	entry = {
		"sessionId": SESSION_ID,
		"timestamp": int(time.time() * 1000),
		"location": location,
		"message": message,
		"data": data or {},
		"hypothesisId": hypothesis_id,
		"runId": run_id,
	}
	try:
		with open(LOG_PATH, "a", encoding="utf-8") as handle:
			handle.write(json.dumps(entry, default=str) + "\n")
	except OSError:
		pass

	if is_debug_enabled():
		payload = f" {data}" if data else ""
		print(f"[DEBUG] {location}: {message}{payload}", flush=True)
