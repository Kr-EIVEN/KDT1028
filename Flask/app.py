# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import mysql.connector, os, uuid, pathlib
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path="/static", static_folder="static")
CORS(app, resources={r"/api/*": {"origins": "*"}})  # 개발용 전체 허용

DB = dict(
    host=os.getenv("MYSQL_HOST","127.0.0.1"),
    user=os.getenv("MYSQL_USER","root"),
    password=os.getenv("MYSQL_PW","your_password"),
    database=os.getenv("MYSQL_DB","picshow"),
)

def conn():
    return mysql.connector.connect(**DB)

# 2-1) 특정 사용자 이미지 목록
@app.get("/api/users/<int:user_id>/images")
def list_user_images(user_id: int):
    limit  = int(request.args.get("limit", 40))
    offset = int(request.args.get("offset", 0))
    q = """
      SELECT id, user_id, filename, url_path, created_at
      FROM user_images
      WHERE user_id=%s
      ORDER BY created_at DESC
      LIMIT %s OFFSET %s
    """
    c = conn(); cur = c.cursor(dictionary=True)
    cur.execute(q, (user_id, limit, offset))
    rows = cur.fetchall()
    cur.close(); c.close()

    # 프론트에서 바로 쓰도록 절대 URL도 함께 제공
    base = request.host_url.rstrip("/")
    for r in rows:
        r["image_url"] = f'{base}{r["url_path"]}'
    return jsonify(rows)

ALLOWED = {"png","jpg","jpeg","webp","gif"}

# 2-2) 업로드 (multipart/form-data)
@app.post("/api/users/<int:user_id>/images")
def upload_user_image(user_id: int):
    file = request.files.get("file")
    if not file:
        return {"error":"no file"}, 400
    ext = file.filename.rsplit(".",1)[-1].lower()
    if ext not in ALLOWED:
        return {"error":"bad type"}, 400

    os.makedirs(f"static/uploads/{user_id}", exist_ok=True)
    fn = secure_filename(f"{uuid.uuid4().hex}.{ext}")
    save_path = pathlib.Path(f"static/uploads/{user_id}")/fn
    file.save(save_path)

    url_path = f"/static/uploads/{user_id}/{fn}"

    c = conn(); cur = c.cursor()
    cur.execute(
        "INSERT INTO user_images(user_id, filename, url_path) VALUES(%s,%s,%s)",
        (user_id, fn, url_path)
    )
    c.commit()
    img_id = cur.lastrowid
    cur.close(); c.close()

    full = request.host_url.rstrip("/") + url_path
    return {"ok": True, "id": img_id, "image_url": full, "url_path": url_path}
