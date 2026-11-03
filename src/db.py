"""SQLite state shared by API and worker containers (WAL mode, mounted volume)."""
import json
import os
import sqlite3
import time
from pathlib import Path

DB = os.getenv("DB_PATH", "data/truthlens.db")
SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, filename TEXT, caption TEXT, status TEXT,
  report TEXT, created REAL, finished REAL, phash TEXT, family TEXT, review_needed INT DEFAULT 0);
CREATE TABLE IF NOT EXISTS reviews(job_id TEXT PRIMARY KEY, label TEXT, note TEXT, ts REAL);
"""


def conn():
    Path(DB).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript(SCHEMA)
    return c


def create_job(job_id, filename, caption):
    with conn() as c:
        c.execute("INSERT INTO jobs(id,filename,caption,status,created) VALUES(?,?,?,?,?)",
                  (job_id, filename, caption, "queued", time.time()))


def finish_job(job_id, status, report, phash=None, family=None):
    review = int(bool(report) and report.get("fusion", {}).get("verdict") == "Uncertain")
    with conn() as c:
        c.execute("UPDATE jobs SET status=?,report=?,finished=?,phash=?,family=?,review_needed=? WHERE id=?",
                  (status, json.dumps(report), time.time(), phash, family, review, job_id))


def _row(r):
    d = dict(r)
    d["report"] = json.loads(d["report"]) if d.get("report") else None
    return d


def get_job(job_id):
    with conn() as c:
        r = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _row(r) if r else None


def list_jobs(limit=50):
    with conn() as c:
        return [_row(r) for r in c.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,))]


def hashes():
    with conn() as c:
        return [(r["id"], r["phash"]) for r in c.execute(
            "SELECT id,phash FROM jobs WHERE phash IS NOT NULL ORDER BY created")]


def family_size(family):
    with conn() as c:
        return c.execute("SELECT COUNT(*) FROM jobs WHERE family=?", (family,)).fetchone()[0]


def review_queue():
    with conn() as c:
        return [_row(r) for r in c.execute(
            "SELECT j.* FROM jobs j LEFT JOIN reviews r ON r.job_id=j.id "
            "WHERE j.review_needed=1 AND r.job_id IS NULL ORDER BY j.created DESC")]


def add_review(job_id, label, note):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO reviews VALUES(?,?,?,?)", (job_id, label, note, time.time()))


def scores():
    with conn() as c:
        return [json.loads(r["report"])["fusion"]["p_fake"] for r in c.execute(
            "SELECT report FROM jobs WHERE status='done' ORDER BY created")
            if json.loads(r["report"]).get("fusion", {}).get("p_fake") is not None]
