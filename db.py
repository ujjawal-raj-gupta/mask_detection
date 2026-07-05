import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "screening_log.db")

# person_id -> datetime of the last write, used to debounce repeat logging
# for a person who keeps standing in frame.
_last_logged = {}


@contextmanager
def get_conn():
	conn = sqlite3.connect(DB_PATH)
	conn.row_factory = sqlite3.Row
	try:
		yield conn
		conn.commit()
	finally:
		conn.close()


def init_db():
	with get_conn() as conn:
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS entries (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				timestamp TEXT NOT NULL,
				person_id INTEGER,
				mask_status TEXT NOT NULL,
				temperature REAL,
				temp_status TEXT,
				snapshot_path TEXT
			)
			"""
		)


def log_entry(mask_status, temperature=None, person_id=None,
		snapshot_path=None, fever_threshold_c=37.5):
	temp_status = None
	if temperature is not None:
		temp_status = "Fever" if temperature >= fever_threshold_c else "Normal"

	with get_conn() as conn:
		conn.execute(
			"""INSERT INTO entries
			   (timestamp, person_id, mask_status, temperature, temp_status, snapshot_path)
			   VALUES (?, ?, ?, ?, ?, ?)""",
			(datetime.now().isoformat(timespec="seconds"),
			 person_id, mask_status, temperature, temp_status, snapshot_path),
		)
	return temp_status


def fetch_recent(limit=50):
	with get_conn() as conn:
		rows = conn.execute(
			"SELECT * FROM entries ORDER BY id DESC LIMIT ?", (limit,)
		).fetchall()
	return [dict(row) for row in rows]


def fetch_today_stats():
	today = date.today().isoformat()
	with get_conn() as conn:
		total = conn.execute(
			"SELECT COUNT(*) c FROM entries WHERE timestamp LIKE ?", (f"{today}%",)
		).fetchone()["c"]
		masked = conn.execute(
			"SELECT COUNT(*) c FROM entries WHERE timestamp LIKE ? AND mask_status = 'Mask'",
			(f"{today}%",),
		).fetchone()["c"]
		fevers = conn.execute(
			"SELECT COUNT(*) c FROM entries WHERE timestamp LIKE ? AND temp_status = 'Fever'",
			(f"{today}%",),
		).fetchone()["c"]

	compliance = round((masked / total) * 100, 1) if total else 0
	return {"total": total, "compliance_pct": compliance, "fevers": fevers}


def _last_logged_at(person_id):
	"""Most recent log time for a tracker ID, from cache or the database.

	The in-memory cache is the hot path (checked every frame). On the first
	sighting of an ID in a fresh process, we fall back to the database so the
	cooldown survives restarts.
	"""
	cached = _last_logged.get(person_id)
	if cached is not None:
		return cached

	with get_conn() as conn:
		row = conn.execute(
			"SELECT timestamp FROM entries WHERE person_id = ? ORDER BY id DESC LIMIT 1",
			(person_id,),
		).fetchone()

	if row is None:
		return None

	try:
		return datetime.fromisoformat(row["timestamp"])
	except (TypeError, ValueError):
		return None


def should_log(person_id, cooldown_seconds=8):
	"""Return True at most once per cooldown window for a given tracker ID.

	Backed by the database so the cooldown persists across process restarts.
	"""
	now = datetime.now()
	last = _last_logged_at(person_id)
	if last is None or (now - last).total_seconds() >= cooldown_seconds:
		_last_logged[person_id] = now
		return True
	return False
