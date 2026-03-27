"""
app.py - May Cham Cong Web Server (Flask + HTTPS)
- API thu thap khuon mat qua camera trinh duyet
- Nhan dien khuon mat voi co che voting nhieu frame (chong nham lan)
- Quan ly nhan vien, bao cao, check-in/out
"""

import os
import sys
import cv2
import numpy as np
import base64
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, redirect
import database
import face_module
from utils import lay_gio_hien_tai, lay_ngay_hien_tai, dinh_dang_hien_thi

# Fix loi UnicodeEncodeError khi print tren Windows Terminal
sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)

# Khoi tao DB khi chay
database.tao_bang()

# Tai truoc model AI - se reload sau khi huan luyen lai
recognizer = face_module.load_mo_hinh()
face_cascade = cv2.CascadeClassifier(face_module.FACE_CASCADE_PATH)

# ------------------------------------------------------------------
# Co che VOTING nhieu frame: Moi session cua trinh duyet se co
# bien dem so lan nhan dien duoc cung 1 ID. Chi xac nhan khi dat >= 3.
# ------------------------------------------------------------------
recognition_votes = defaultdict(int)  # {session_token: {id: count}}
recognition_history = defaultdict(list)  # {session_token: [id1, id2, ...]}
VOTE_THRESHOLD = 3  # So frame can de xac nhan


# ==============================================================
# ROUTES - TRANG WEB
# ==============================================================

@app.route('/')
def home():
    """Trang cham cong - Camera + Check-in/out."""
    return render_template('index.html')


@app.route('/nhan_vien')
def quan_ly_nhan_vien():
    """Trang quan ly nhan vien."""
    ds_nhan_vien = database.lay_tat_ca_nhan_vien()
    # Them so luong anh mat da co cua moi nhan vien
    ds_voi_anh = []
    for nv in ds_nhan_vien:
        so_anh = _dem_anh_khuon_mat(nv[0])  # nv[0] = id, nv[1] = ma_nv
        ds_voi_anh.append({
            "id": nv[0],
            "ma_nv": nv[1],
            "ten_nv": nv[2],
            "phong_ban": nv[3],
            "so_anh": so_anh
        })
    return render_template('nhan_vien.html', nhan_viens=ds_voi_anh)


@app.route('/bao_cao')
def trang_bao_cao():
    """Trang bao cao."""
    return render_template('bao_cao.html')


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def _dem_anh_khuon_mat(nv_id: int) -> int:
    """Dem so anh mau da co cua mot nhan vien theo ID so."""
    faces_dir = face_module.DATA_FACES_DIR
    if not os.path.exists(faces_dir):
        return 0
    return len([f for f in os.listdir(faces_dir) if f.startswith(f"User.{nv_id}.")])


def _xu_ly_anh_base64(img_b64: str):
    """
    Giai ma chuoi base64 thanh anh OpenCV.
    Chi tra ve anh xam NGUYEN BAN, khong ap dung equalize (de Haar Cascade hoat dong chinh xac).
    Returns: (img_color, img_gray) hoac (None, None) neu loi
    """
    try:
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(img_b64), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, None
        # Tra ve anh xam goc - KHONG equalizeHist de Haar Cascade phat hien khuon mat tot hon
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray
    except Exception as e:
        print(f"[ERROR] _xu_ly_anh_base64: {e}")
        return None, None


# ==============================================================
# API - NHAN VIEN
# ==============================================================

@app.route('/api/them_nhan_vien', methods=['POST'])
def api_them_nhan_vien():
    """API them nhan vien moi."""
    data = request.json
    ma_nv = data.get('ma_nv', '').strip().upper()
    ten_nv = data.get('ten_nv', '').strip()
    phong_ban = data.get('phong_ban', '').strip()

    if not ma_nv or not ten_nv:
        return jsonify({"thanh_cong": False, "thong_bao": "Thieu ma NV hoac ten NV"})

    kq = database.them_nhan_vien(ma_nv, ten_nv, phong_ban)
    return jsonify(kq)


@app.route('/api/kiem_tra_khuon_mat/<ma_nv>')
def api_kiem_tra_khuon_mat(ma_nv):
    """API kiem tra so anh mat da co cua nhan vien."""
    nv = database.tim_nhan_vien(ma_nv)
    if not nv:
        return jsonify({"thanh_cong": False, "so_anh": 0})
    so_anh = _dem_anh_khuon_mat(nv[0])
    return jsonify({"thanh_cong": True, "so_anh": so_anh, "nv_id": nv[0]})


# ==============================================================
# API - THU THAP KHUON MAT QUA CAMERA TRINH DUYET
# ==============================================================

@app.route('/api/thu_thap_anh', methods=['POST'])
def api_thu_thap_anh():
    """
    Nhan anh base64 tu camera trinh duyet, phat hien khuon mat,
    luu vao the muc data/faces. Tra ve so luong anh da luu.
    """
    data = request.json
    ma_nv = data.get('ma_nv', '').strip().upper()
    img_b64 = data.get('image', '')

    if not ma_nv or not img_b64:
        return jsonify({"thanh_cong": False, "thong_bao": "Thieu du lieu"})

    # Kiem tra nhan vien ton tai
    nv = database.tim_nhan_vien(ma_nv)
    if not nv:
        return jsonify({"thanh_cong": False, "thong_bao": f"Khong tim thay NV {ma_nv}"})

    nv_id = nv[0]  # ID so nguyen cua nhan vien

    img, gray = _xu_ly_anh_base64(img_b64)
    if img is None:
        return jsonify({"thanh_cong": False, "thong_bao": "Loi giai ma anh"})

    # Phat hien khuon mat - dung tham so mac dinh de bat duoc khuon mat tu nhieu goc/khoang cach
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    if len(faces) == 0:
        return jsonify({"thanh_cong": False, "thong_bao": "Khong tim thay khuon mat trong anh"})

    # Lay khuon mat lon nhat
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]

    # Dem so anh hien tai de dat ten file lien tiep
    so_anh_hien_tai = _dem_anh_khuon_mat(nv_id)
    file_path = os.path.join(face_module.DATA_FACES_DIR, f"User.{nv_id}.{so_anh_hien_tai + 1}.jpg")

    # Resize anh khuon mat ve kich thuoc chuan 100x100
    face_resized = cv2.resize(face_roi, (100, 100))
    cv2.imwrite(file_path, face_resized)

    so_anh_moi = so_anh_hien_tai + 1
    print(f"[THU THAP] NV={ma_nv} ID={nv_id} - Da luu {so_anh_moi} anh")

    return jsonify({
        "thanh_cong": True,
        "so_anh": so_anh_moi,
        "thong_bao": f"Da luu anh thu {so_anh_moi}"
    })


# ==============================================================
# API - HUAN LUYEN AI
# ==============================================================

@app.route('/api/huan_luyen', methods=['POST'])
def api_huan_luyen():
    """Huan luyen lai mo hinh AI va tai lai recognizer trong bo nho."""
    kq = face_module.huan_luyen_mo_hinh()
    if kq["thanh_cong"]:
        global recognizer
        recognizer = face_module.load_mo_hinh()
        print("[AI] Mo hinh da duoc huan luyen va tai lai thanh cong")
    return jsonify(kq)


# ==============================================================
# API - NHAN DIEN KHUON MAT (VOI VOTING)
# ==============================================================

@app.route('/api/nhan_dien', methods=['POST'])
def api_nhan_dien():
    """
    Nhan anh base64 tu trinh duyet, phat hien + nhan dien khuon mat.
    Su dung co che voting: can >= 3 frame lien tiep cung mot ID moi tra ket qua.
    """
    if not recognizer:
        return jsonify({"thanh_cong": False, "thong_bao": "Chua huan luyen AI"})

    data = request.json
    if not data or 'image' not in data:
        return jsonify({"thanh_cong": False, "thong_bao": "Khong co du lieu anh"})

    # Token de phan biet cac session / trinh duyet khac nhau
    session_token = data.get('session_token', 'default')

    try:
        img, gray = _xu_ly_anh_base64(data['image'])
        if img is None:
            return jsonify({"thanh_cong": False, "thong_bao": "Loi giai ma anh"})

        # Phat hien khuon mat - tham so noi long de nhan dien duoc qua camera dien thoai
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        if len(faces) == 0:
            # Reset lich su voting khi mat khuon mat
            recognition_history[session_token].clear()
            return jsonify({"thanh_cong": False, "thong_bao": "Khong tim thay khuon mat"})

        # Lay khuon mat lon nhat trong frame
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]

        # Resize ve kich thuoc chuan truoc khi nhan dien
        # Ap dung equalizeHist TREN CROP KHUON MAT (khong phai toan bo frame) - nhat quan voi training
        face_equalized = cv2.equalizeHist(face_roi)
        face_resized = cv2.resize(face_equalized, (100, 100))
        id_detected, confidence = recognizer.predict(face_resized)

        print(f"[AI] Session={session_token[:8]} ID={id_detected} Confidence={confidence:.1f}")

        # Nguong chap nhan: confidence cang nho cang giong nguoi that
        if confidence > 80:
            # Confidence qua cao = khong nhan ra
            recognition_history[session_token].clear()
            return jsonify({"thanh_cong": False, "thong_bao": f"Chua xac dinh (conf={confidence:.0f})"})

        # Them vao lich su voting
        history = recognition_history[session_token]
        history.append(id_detected)

        # Chi giu lai 10 frame cuoi
        if len(history) > 10:
            history.pop(0)

        # Dem so lan xuat hien cua ID nay trong lich su
        vote_count = history.count(id_detected)
        print(f"[VOTING] ID={id_detected} Votes={vote_count}/{VOTE_THRESHOLD}")

        if vote_count >= VOTE_THRESHOLD:
            # Du vote: tra ve ket qua nhan dien
            ma_nv = database.lay_ma_nv_tu_id(id_detected)
            if ma_nv:
                nv = database.tim_nhan_vien(ma_nv)
                return jsonify({
                    "thanh_cong": True,
                    "ma_nv": ma_nv,
                    "ten_nv": nv[2] if nv else "",
                    "phong_ban": nv[3] if nv else "",
                    "confidence": round(confidence, 1)
                })

        # Chua du vote, bao cho frontend biet dang trong qua trinh xac nhan
        return jsonify({
            "thanh_cong": False,
            "dang_xac_nhan": True,
            "thong_bao": f"Dang xac nhan... ({vote_count}/{VOTE_THRESHOLD})"
        })

    except Exception as e:
        print(f"[ERROR] api_nhan_dien: {e}")
        return jsonify({"thanh_cong": False, "thong_bao": f"Loi: {str(e)}"})


# ==============================================================
# API - CHAM CONG (CHECK-IN / CHECK-OUT)
# ==============================================================

@app.route('/api/cham_cong', methods=['POST'])
def api_cham_cong():
    """Check-in hoac Check-out cho nhan vien."""
    data = request.json
    ma_nv = data.get('ma_nv', '').strip().upper()
    phuong_thuc = data.get('phuong_thuc', 'in')  # 'in' hoac 'out'
    session_token = data.get('session_token', 'default')

    if not ma_nv:
        return jsonify({"thanh_cong": False, "thong_bao": "Chua co ma nhan vien"})

    if phuong_thuc == 'in':
        kq = database.check_in(ma_nv)
    else:
        kq = database.check_out(ma_nv)

    # Reset voting history sau khi cham cong thanh cong
    if kq.get("thanh_cong"):
        recognition_history[session_token].clear()

    return jsonify(kq)


# ==============================================================
# API - BAO CAO
# ==============================================================

@app.route('/api/bao_cao_data')
def api_bao_cao_data():
    """API lay du lieu bao cao."""
    ma = request.args.get('ma_nv', '').strip().upper()
    tu = request.args.get('tu_ngay', lay_ngay_hien_tai())
    den = request.args.get('den_ngay', lay_ngay_hien_tai())

    rows = database.lay_cham_cong_tu_den(tu, den, ma if ma else None)

    results = []
    tong_tat_ca = 0.0
    for r in rows:
        tong_gio = r[5] if r[5] else 0.0
        tong_tat_ca += tong_gio
        results.append({
            "ma_nv": r[0], "ten_nv": r[1], "ngay": r[2],
            "gio_vao": r[3] or "", "gio_ra": r[4] or "",
            "tong_gio_str": dinh_dang_hien_thi(tong_gio)
        })

    return jsonify({
        "thanh_cong": True,
        "data": results,
        "summary": f"Tong so ban ghi: {len(rows)} | Tong gio lam viec: {round(tong_tat_ca, 2)} gio"
    })


# ==============================================================
# MAIN - KHOI CHAY SERVER
# ==============================================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("WEB SERVER STARTED - HTTPS MODE")
    print("Run ipconfig to find your IPv4. E.g: 192.168.1.xxx")
    print("On your PHONE use Chrome and go to: https://<your_ipv4>:5000")
    print("If browser warns Not Secure: click Advanced then Continue")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, ssl_context="adhoc")
