import math
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for
from PIL import Image, ImageStat

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "app.db"
UPLOAD_DIR = BASE_DIR / "uploads"
NORMALIZED_DIR = UPLOAD_DIR / "normalized"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_UPLOAD_MB = 10
TARGET_SIZE = (1024, 1024)
SIMILARITY_THRESHOLD = 0.92

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                normalized_filename TEXT NOT NULL,
                format TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                quality_score REAL NOT NULL,
                embedding TEXT NOT NULL,
                tag TEXT NOT NULL,
                faces_detected INTEGER NOT NULL,
                event_key TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS clusters (
                id TEXT PRIMARY KEY,
                centroid TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cluster_members (
                cluster_id TEXT NOT NULL,
                image_id TEXT NOT NULL,
                PRIMARY KEY (cluster_id, image_id)
            );
            """
        )


init_db()


def serialize_vector(vector: np.ndarray) -> str:
    return ",".join(f"{v:.6f}" for v in vector.tolist())


def deserialize_vector(value: str) -> np.ndarray:
    return np.array([float(v) for v in value.split(",")], dtype=np.float32)


def compute_embedding(image: Image.Image) -> np.ndarray:
    small = image.resize((64, 64))
    np_image = np.asarray(small, dtype=np.float32) / 255.0
    histogram = np.histogram(np_image, bins=12, range=(0, 1))[0].astype(np.float32)
    histogram = histogram / (np.linalg.norm(histogram) + 1e-8)
    return histogram


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def quality_score(image: Image.Image) -> float:
    stat = ImageStat.Stat(image.convert("L"))
    variance = stat.var[0]
    return min(1.0, variance / 5000.0)


def detect_faces(image: Image.Image) -> int:
    width, height = image.size
    if width * height < 200 * 200:
        return 0
    ratio = width / max(height, 1)
    if 0.8 <= ratio <= 1.4:
        return 1
    return 0


def pick_tag(image: Image.Image) -> str:
    np_image = np.asarray(image.resize((8, 8)), dtype=np.float32)
    avg = np_image.mean(axis=(0, 1))
    r, g, b = avg
    if r > g and r > b:
        return "warm"
    if b > r and b > g:
        return "cool"
    return "neutral"


def assign_event_key(timestamp: datetime) -> str:
    return timestamp.strftime("%Y-%m-%d-%H")


def store_image(image: Image.Image, filename: str, normalized_name: str) -> None:
    image.save(UPLOAD_DIR / filename)
    normalized_path = NORMALIZED_DIR / normalized_name
    image.save(normalized_path)


def find_or_create_cluster(embedding: np.ndarray) -> str:
    with get_db() as conn:
        rows = conn.execute("SELECT id, centroid FROM clusters").fetchall()
        best_id = None
        best_score = 0.0
        for row in rows:
            centroid = deserialize_vector(row["centroid"])
            score = cosine_similarity(embedding, centroid)
            if score > best_score:
                best_score = score
                best_id = row["id"]
        if best_id and best_score >= SIMILARITY_THRESHOLD:
            return best_id
        cluster_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO clusters (id, centroid, created_at) VALUES (?, ?, ?)",
            (cluster_id, serialize_vector(embedding), datetime.utcnow().isoformat()),
        )
        return cluster_id


def update_cluster_centroid(cluster_id: str) -> None:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT images.embedding FROM images "
            "JOIN cluster_members ON images.id = cluster_members.image_id "
            "WHERE cluster_members.cluster_id = ?",
            (cluster_id,),
        ).fetchall()
        embeddings = [deserialize_vector(row["embedding"]) for row in rows]
        if not embeddings:
            return
        centroid = np.mean(np.stack(embeddings, axis=0), axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        conn.execute(
            "UPDATE clusters SET centroid = ? WHERE id = ?",
            (serialize_vector(centroid), cluster_id),
        )


def insert_image_record(
    image_id: str,
    filename: str,
    normalized_filename: str,
    image: Image.Image,
    embedding: np.ndarray,
    tag: str,
    faces_detected: int,
    event_key: str,
    quality: float,
) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO images
            (id, filename, normalized_filename, format, width, height, created_at, quality_score, embedding, tag, faces_detected, event_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                filename,
                normalized_filename,
                image.format or "UNKNOWN",
                image.width,
                image.height,
                datetime.utcnow().isoformat(),
                quality,
                serialize_vector(embedding),
                tag,
                faces_detected,
                event_key,
            ),
        )


def ensure_valid_image(image: Image.Image) -> Tuple[bool, str]:
    if image.format not in ALLOWED_FORMATS:
        return False, "Unsupported file format."
    if image.width < 64 or image.height < 64:
        return False, "Image is too small."
    return True, ""


@app.route("/")
def index():
    with get_db() as conn:
        images = conn.execute(
            "SELECT id, normalized_filename, tag, quality_score, faces_detected, event_key "
            "FROM images ORDER BY created_at DESC LIMIT 12"
        ).fetchall()
    return render_template("index.html", images=images)


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return redirect(url_for("index"))
    file = request.files["image"]
    if not file or file.filename == "":
        return redirect(url_for("index"))
    image = Image.open(file.stream)
    is_valid, message = ensure_valid_image(image)
    if not is_valid:
        return render_template("index.html", error=message, images=[])

    normalized_image = image.copy().convert("RGB")
    normalized_image.thumbnail(TARGET_SIZE)

    quality = quality_score(normalized_image)
    embedding = compute_embedding(normalized_image)
    faces = detect_faces(normalized_image)
    tag = pick_tag(normalized_image)

    image_id = str(uuid.uuid4())
    ext = "jpg" if image.format == "JPEG" else "png"
    filename = f"{image_id}.{ext}"
    normalized_filename = f"normalized_{image_id}.{ext}"

    store_image(normalized_image, filename, normalized_filename)

    event_key = assign_event_key(datetime.utcnow())
    insert_image_record(
        image_id,
        filename,
        normalized_filename,
        image,
        embedding,
        tag,
        faces,
        event_key,
        quality,
    )

    if faces:
        cluster_id = find_or_create_cluster(embedding)
        with get_db() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO cluster_members (cluster_id, image_id) VALUES (?, ?)",
                (cluster_id, image_id),
            )
        update_cluster_centroid(cluster_id)

    return redirect(url_for("index"))


@app.route("/uploads/<path:filename>")
def serve_upload(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/search")
def search():
    query = request.args.get("q", "").strip().lower()
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, normalized_filename, tag, quality_score, faces_detected, event_key "
            "FROM images WHERE tag LIKE ? OR filename LIKE ? ORDER BY created_at DESC",
            (f"%{query}%", f"%{query}%"),
        ).fetchall()
    return render_template("search.html", query=query, images=rows)


@app.route("/clusters")
def clusters():
    with get_db() as conn:
        clusters = conn.execute(
            "SELECT id, created_at FROM clusters ORDER BY created_at DESC"
        ).fetchall()
        members = conn.execute(
            """
            SELECT cluster_members.cluster_id, images.normalized_filename
            FROM cluster_members
            JOIN images ON images.id = cluster_members.image_id
            """
        ).fetchall()
    grouped = {}
    for member in members:
        grouped.setdefault(member["cluster_id"], []).append(member["normalized_filename"])
    return render_template("clusters.html", clusters=clusters, members=grouped)


@app.route("/events")
def events():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT event_key, COUNT(*) as count FROM images GROUP BY event_key ORDER BY event_key DESC"
        ).fetchall()
    return render_template("events.html", events=rows)


@app.route("/api/face-search", methods=["POST"])
def face_search():
    if "image" not in request.files:
        return jsonify({"error": "Missing image"}), 400
    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")
    embedding = compute_embedding(image)
    with get_db() as conn:
        rows = conn.execute("SELECT id, normalized_filename, embedding FROM images").fetchall()
    results: List[Tuple[str, float, str]] = []
    for row in rows:
        score = cosine_similarity(embedding, deserialize_vector(row["embedding"]))
        results.append((row["id"], score, row["normalized_filename"]))
    results.sort(key=lambda item: item[1], reverse=True)
    top = [
        {"id": image_id, "score": round(score, 3), "filename": filename}
        for image_id, score, filename in results[:8]
    ]
    return jsonify({"results": top})


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=False)
