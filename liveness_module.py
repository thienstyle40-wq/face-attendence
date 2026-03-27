"""
liveness_module.py
Mục đích: Module kiểm tra liveness (giống/ảnh thực vs giả mạo) sử dụng thuật toán LBP.
Tác giả: Antigravity Assistant
"""

import cv2
import numpy as np
import os
import pickle

# Nếu thiếu thư viện, cần pip install scikit-image scikit-learn
try:
    from skimage import feature
    from sklearn.svm import SVC
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False

MODEL_PATH = "data/liveness_model.pkl"

class LivenessDetector:
    def __init__(self):
        self.model = None
        self.radius = 3
        self.n_points = 24
        self.load_model()
        
    def extract_lbp(self, image):
        """
        Trích xuất đặc trưng LBP từ ảnh.
        Yêu cầu đầu vào: ảnh xám (grayscale).
        """
        if not _SKIMAGE_AVAILABLE:
            # Fallback nếu không có skimage, nhưng LBP cần skimage để chuẩn nhất
            return None
            
        lbp = feature.local_binary_pattern(image, self.n_points, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.n_points + 3), range=(0, self.n_points + 2))
        
        # Chuẩn hoá histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist

    def train(self, real_faces_dir, fake_faces_dir, verbose=False):
        """
        Huấn luyện mô hình Liveness SVM bằng đặc trưng LBP.
        """
        if not _SKIMAGE_AVAILABLE:
            if verbose: print("Lỗi: Cần cài đặt scikit-image và scikit-learn (pip install scikit-image scikit-learn)")
            return False

        data = []
        labels = []
        
        # Đọc ảnh Real (nhãn 1)
        if os.path.exists(real_faces_dir):
            for f in os.listdir(real_faces_dir):
                if f.endswith(".jpg") or f.endswith(".png"):
                    path = os.path.join(real_faces_dir, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (100, 100))
                        hist = self.extract_lbp(img)
                        if hist is not None:
                            data.append(hist)
                            labels.append(1)
        
        # Đọc ảnh Fake (nhãn 0)
        if os.path.exists(fake_faces_dir):
            for f in os.listdir(fake_faces_dir):
                if f.endswith(".jpg") or f.endswith(".png"):
                    path = os.path.join(fake_faces_dir, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (100, 100))
                        hist = self.extract_lbp(img)
                        if hist is not None:
                            data.append(hist)
                            labels.append(0)

        if len(data) == 0:
            if verbose: print("Không có dữ liệu huấn luyện!")
            return False
            
        if verbose: print(f"Đang huấn luyện với {len(data)} mẫu...")
        
        model = SVC(kernel="linear", probability=True)
        model.fit(data, labels)
        
        # Lưu mô hình
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
            
        self.model = model
        if verbose: print("Đã huấn luyện và lưu liveness mô hình thành công.")
        return True

    def load_model(self):
        """
        Tải mô hình Liveness (nếu có).
        """
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
            except Exception as e:
                print(f"Lỗi khi tải Liveness Model: {e}")
                self.model = None

    def predict(self, face_img):
        """
        Dự đoán ảnh face là Real hay Fake.
        face_img: Ảnh xám khuôn mặt (Grayscale crop vùng có mặt).
        Trả về: True (Real), False (Fake).
        Nếu chưa có mô hình hoặc chưa có thư viện, luôn trả về True (bỏ qua check).
        """
        # Nếu không có sklearn/skimage hoặc không có model đã train, skip pass cho Real
        if not _SKIMAGE_AVAILABLE or self.model is None:
            return True
            
        # Resize về cỡ chuẩn
        face_img = cv2.resize(face_img, (100, 100))
        hist = self.extract_lbp(face_img)
        
        if hist is None:
            return True # Lỗi trích xuất
            
        prediction = self.model.predict([hist])[0]
        prob = self.model.predict_proba([hist])[0]
        
        # Hạ ngưỡng confidence từ 0.6 xuống 0.5 hoặc chỉ cần check prediction để tránh nhận sai mặt thật (false negative)
        if prediction == 1 and prob[1] >= 0.5: 
            return True # Real
        else:
            return False # Fake
