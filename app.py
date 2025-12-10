
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import jwt  # Thư viện để tạo và xác thực JWT
import datetime
from functools import wraps
from flask_cors import CORS
import cv2, json
from ultralytics import YOLO
from main import process
from temp.json_export import *


app = Flask(__name__)

# Security: Use environment variables for secrets and config
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "instance", "user.db"))
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("SQLALCHEMY_DATABASE_URI", f"sqlite:///{db_path}")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", os.urandom(24))
app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_FOLDER", "backend/uploads")
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(minutes=30)

CORS(app, origins=[os.environ.get("CORS_ORIGIN", "http://localhost:5173")], methods=["GET", "POST"], allow_headers=["Content-Type", "Authorization"])

# Ensure upload folder exists
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

# Load YOLO model path from environment or default

YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
try:
    model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    print(f"[ERROR] Failed to load YOLO model from {YOLO_MODEL_PATH}: {e}")
    model = None

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    fullname = db.Column(db.String(80), nullable=False)



# Always create tables if missing, using correct DB path
with app.app_context():
    db_path = os.path.join("backend", "instance", "user.db")
    if not os.path.exists(db_path):
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db.create_all()
    print("Ensured database and tables exist!")
        
        
# Hàm tạo JWT
def create_access_token(user):
    # Security: Token expiration from env or default 60 min
    expiration_minutes = int(os.environ.get("JWT_EXP_MINUTES", 60))
    expiration = datetime.datetime.utcnow() + datetime.timedelta(minutes=expiration_minutes)
    token = jwt.encode(
        {"username": user.username, "exp": expiration, "fullname": user.fullname},
        app.config["SECRET_KEY"],
        algorithm="HS256"
    )
    return token

# Hàm xác thực JWT
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # Lấy token từ header Authorization
        if "Authorization" in request.headers:
            token = request.headers["Authorization"].split(" ")[1]  # Format: Bearer <token>
        if not token:
            return jsonify({"message": "Token is missing!"}), 401

        try:
            # Giải mã token
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            current_user = User.query.get(data["username"])
        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Token has expired!"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "Invalid token!"}), 401

        # Truyền user hiện tại vào hàm được decorate
        return f(current_user, *args, **kwargs)
    return decorated  

def process_video_with_yolo(video_path):
    # Đọc video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file")

    # Lấy thông tin video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Tạo đường dẫn video đã xử lý
    processed_video_path = os.path.join(app.config["UPLOAD_FOLDER"], f"processed_{os.path.basename(video_path)}")
    out = cv2.VideoWriter(processed_video_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (frame_width, frame_height))

    # Xử lý từng frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect vật thể bằng YOLO
        results = model(frame)
        annotated_frame = results[0].plot()  # Vẽ bounding box và nhãn lên frame

        # Ghi frame đã xử lý vào video mới
        out.write(annotated_frame)

    # Giải phóng tài nguyên
    cap.release()
    out.release()

    return processed_video_path

def process_ai_football(video_path):
    path, tracks = process(video_path)
    data = clear(tracks)
    return data, path

# Route đăng nhập
@app.route("/api/login", methods=["POST"])
def login():
    # Lấy dữ liệu từ request
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    # Kiểm tra xem username và password có được cung cấp không
    if not username or not password:
        return jsonify({"message": "Username and password are required!"}), 400

    # Tìm user trong cơ sở dữ liệu
    user = User.query.filter_by(username=username).first()

    # Kiểm tra user và mật khẩu
    if user and check_password_hash(user.password, password):
        # Tạo access_token
        access_token = create_access_token(user)
        return jsonify({
            "message": "Login successful!",
            "token": access_token
        }), 200
    else:
        return jsonify({"message": "Invalid username or password!"}), 401
    
# Route bảo mật (yêu cầu token)
@app.route("/api/upload-video", methods=["POST"])
@token_required

def allowed_file(filename):
    allowed_extensions = {"mp4", "avi", "mov", "mkv"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

@app.route("/api/upload-video", methods=["POST"])
@token_required
def upload_video():
    # Check for file
    if "video" not in request.files:
        return jsonify({"message": "No video file provided!"}), 400

    video_file = request.files["video"]

    # Check filename
    if video_file.filename == "":
        return jsonify({"message": "No selected file!"}), 400

    # Security: Sanitize filename and check extension
    filename = secure_filename(video_file.filename)
    if not allowed_file(filename):
        return jsonify({"message": "Invalid file type!"}), 400

    # Save with unique name to prevent overwrite
    unique_filename = f"{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{filename}"
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    video_file.save(video_path)

    # Process video
    data, processed_video_path = process_ai_football(video_path)

    return jsonify({
        "message": "Video uploaded successfully!",
        "main_video": processed_video_path[0],
        "circle_video": processed_video_path[1],
        "voronoi_video": processed_video_path[2],
        "line_video": processed_video_path[3],
        "data": data
    }), 200


# Public video
@app.route("/uploads/<filename>")
def serve_video(filename):
    return send_from_directory("uploads", filename, mimetype="video/mp4")


# Homepage route
@app.route("/")
def home():
    return "<h2>Football Tracking and Analysis API is running.<br>Use /api/login or /api/upload-video endpoints.</h2>"

if __name__ == "__main__":
    app.run(debug=False)
