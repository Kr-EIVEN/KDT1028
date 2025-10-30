from flask import Flask, jsonify, request
from flask_cors import CORS
import mysql.connector, os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

DB = dict(
    host="127.0.0.1",
    user="root",
    password="1111",   # ✅ 본인 MySQL 비번
    database="picshow",
)

def conn():
    return mysql.connector.connect(**DB)

# -------------------
#  회원가입 API
# -------------------
@app.post("/api/auth/signup")
def signup():
    data = request.get_json()
    email = data.get("email")
    pw = data.get("password")
    name = data.get("nickname")

    if not email or not pw or not name:
        return jsonify({"error": "필수 항목 누락"}), 400

    c = conn(); cur = c.cursor(dictionary=True)
    cur.execute("SELECT id FROM users WHERE email=%s", (email,))
    if cur.fetchone():
        cur.close(); c.close()
        return jsonify({"error": "이미 가입된 이메일입니다."}), 409

    hashed = generate_password_hash(pw)
    cur.execute(
        "INSERT INTO users (email, password_hash, name) VALUES (%s, %s, %s)",
        (email, hashed, name),
    )
    c.commit()
    uid = cur.lastrowid
    cur.close(); c.close()

    return jsonify({"ok": True, "user_id": uid, "email": email, "name": name})

# -------------------
#  로그인 API
# -------------------
@app.post("/api/auth/login")
def login():
    data = request.get_json()
    email = data.get("email")
    pw = data.get("password")

    c = conn(); cur = c.cursor(dictionary=True)
    cur.execute("SELECT id, password_hash, name FROM users WHERE email=%s", (email,))
    user = cur.fetchone()
    cur.close(); c.close()

    if not user:
        return jsonify({"error": "존재하지 않는 이메일입니다."}), 404
    if not check_password_hash(user["password_hash"], pw):
        return jsonify({"error": "비밀번호가 올바르지 않습니다."}), 401

    return jsonify({
        "ok": True,
        "user_id": user["id"],
        "email": email,
        "nickname": user["name"],
    })

# -------------------
#  서버 헬스체크
# -------------------
@app.get("/api/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
