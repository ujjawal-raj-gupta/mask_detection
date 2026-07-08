import csv
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "screening_log.db")
CSV_PATH = os.path.join(os.path.dirname(__file__), "screening_log.csv")
CSV_HEADERS = [
	"id", "timestamp", "person_id", "mask_status",
	"temperature", "temp_status", "snapshot_path",
]

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


def _entry_to_csv_row(row):
	return [
		row["id"],
		row["timestamp"],
		row["person_id"] if row["person_id"] is not None else "",
		row["mask_status"],
		row["temperature"] if row["temperature"] is not None else "",
		row["temp_status"] if row["temp_status"] is not None else "",
		row["snapshot_path"] if row["snapshot_path"] is not None else "",
	]


def _ensure_csv_header():
	if os.path.isfile(CSV_PATH):
		return

	with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as csv_file:
		csv.writer(csv_file).writerow(CSV_HEADERS)


def _append_csv_row(entry_id, timestamp, person_id, mask_status,
		temperature, temp_status, snapshot_path):
	_ensure_csv_header()
	with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as csv_file:
		csv.writer(csv_file).writerow([
			entry_id,
			timestamp,
			person_id if person_id is not None else "",
			mask_status,
			temperature if temperature is not None else "",
			temp_status if temp_status is not None else "",
			snapshot_path if snapshot_path is not None else "",
		])


def _rewrite_csv_from_db():
	with get_conn() as conn:
		rows = conn.execute(
			"SELECT * FROM entries ORDER BY id ASC"
		).fetchall()

	with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(CSV_HEADERS)
		for row in rows:
			writer.writerow(_entry_to_csv_row(row))


def _sync_csv_if_needed():
	with get_conn() as conn:
		db_count = conn.execute("SELECT COUNT(*) c FROM entries").fetchone()["c"]

	if db_count == 0:
		_ensure_csv_header()
		return

	csv_rows = 0
	if os.path.isfile(CSV_PATH):
		with open(CSV_PATH, "r", encoding="utf-8") as csv_file:
			csv_rows = max(0, sum(1 for _ in csv_file) - 1)

	if csv_rows < db_count:
		_rewrite_csv_from_db()


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
	_sync_csv_if_needed()


def log_entry(mask_status, temperature=None, person_id=None,
		snapshot_path=None, fever_threshold_c=37.5):
	temp_status = None
	if temperature is not None:
		temp_status = "Fever" if temperature >= fever_threshold_c else "Normal"

	timestamp = datetime.now().isoformat(timespec="seconds")
	with get_conn() as conn:
		cursor = conn.execute(
			"""INSERT INTO entries
			   (timestamp, person_id, mask_status, temperature, temp_status, snapshot_path)
			   VALUES (?, ?, ?, ?, ?, ?)""",
			(timestamp, person_id, mask_status, temperature, temp_status, snapshot_path),
		)
		entry_id = cursor.lastrowid

	_append_csv_row(
		entry_id, timestamp, person_id, mask_status,
		temperature, temp_status, snapshot_path)
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


def _remove_snapshot(snapshot_path):
	if not snapshot_path:
		return

	filepath = os.path.join(os.path.dirname(__file__), snapshot_path.replace("/", os.sep))
	if os.path.isfile(filepath):
		os.remove(filepath)


def delete_entry(entry_id):
	with get_conn() as conn:
		row = conn.execute(
			"SELECT snapshot_path FROM entries WHERE id = ?", (entry_id,)
		).fetchone()
		if row is None:
			return False

		conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))

	_remove_snapshot(row["snapshot_path"] if row else None)
	_rewrite_csv_from_db()
	return True


def delete_all_entries():
	with get_conn() as conn:
		rows = conn.execute(
			"SELECT snapshot_path FROM entries WHERE snapshot_path IS NOT NULL"
		).fetchall()
		deleted = conn.execute("DELETE FROM entries").rowcount

	for row in rows:
		_remove_snapshot(row["snapshot_path"])

	_last_logged.clear()
	_rewrite_csv_from_db()
	return deleted


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
