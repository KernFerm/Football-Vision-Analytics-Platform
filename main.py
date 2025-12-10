# Security: Import only required modules, avoid wildcard imports
from temp.process_frame import read_video, write_video, save_pkl
from tracking.tracking import Tracker
from camera_movement.camera_movement import CameraMovement
from transformer.transformer import Transformer
from speed_and_distance.speed_and_distance import SpeedAndDistance
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
import os
import pickle
import sys
import numpy as np



def process(video_path, model_path=None, model_keypoints_path=None, load_pkl=True):
    """
    Process a football match video for tracking and analytics.
    All paths and sensitive config should be set via environment variables for security.
    """
    # Security: Use environment variables for all paths
    model_path = model_path or os.environ.get("MODEL_PATH", "models/yolov8n.pt")
    model_keypoints_path = model_keypoints_path or os.environ.get("MODEL_KEYPOINTS_PATH", "models/key_points_pitch_ver2.pt")
    outputs_dir = os.environ.get("OUTPUTS_DIR", "./outputs/")
    os.makedirs(outputs_dir, exist_ok=True)

    # Input validation: check video path
    if not os.path.isfile(video_path) or not video_path.lower().endswith(('.mp4', '.avi', '.mov')):
        raise ValueError(f"Invalid or missing video file: {video_path}")

    try:
        frames = read_video(video_path)
    except Exception as e:
        print(f"[ERROR] Failed to read video: {e}")
        sys.exit(1)

    if not frames:
        print(f"[WARNING] No frames extracted from video: {video_path}. Skipping processing.")
        return [], {}
    try:
        tracker = Tracker(model_path, model_keypoints_path)
        camera_movement = CameraMovement(frames[0])
        speed_and_distance = SpeedAndDistance()
    except Exception as e:
        print(f"[ERROR] Failed to initialize models: {e}")
        sys.exit(1)

    tracks_pkl = os.path.join(outputs_dir, "tracks.pkl")
    camera_movement_pkl = os.path.join(outputs_dir, "camera_movement_frames.pkl")
    team_ball_control_pkl = os.path.join(outputs_dir, "team_ball_control.pkl")

    # Security: Use try/except for file operations
    if load_pkl and os.path.exists(tracks_pkl):
        try:
            with open(tracks_pkl, "rb") as f:
                tracks = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load tracks.pkl: {e}")
            tracks = tracker.get_object_tracks(frames)
            tracker.add_position_to_tracks(tracks)
    else:
        tracks = tracker.get_object_tracks(frames)
        tracker.add_position_to_tracks(tracks)

    if load_pkl and os.path.exists(camera_movement_pkl):
        try:
            with open(camera_movement_pkl, "rb") as f:
                camera_movement_frames = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load camera_movement_frames.pkl: {e}")
            camera_movement_frames = camera_movement.get_camera_movement(frames)
            camera_movement.add_camera_movement_to_tracks(tracks, camera_movement_frames)
    else:
        camera_movement_frames = camera_movement.get_camera_movement(frames)
        camera_movement.add_camera_movement_to_tracks(tracks, camera_movement_frames)

    if not load_pkl:
        transformer = Transformer()
        transformer.add_transformed_point(tracks)
        tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])
        speed_and_distance.add_speed_and_distance_to_tracks(tracks)

    if load_pkl and os.path.exists(team_ball_control_pkl):
        try:
            with open(team_ball_control_pkl, "rb") as f:
                team_ball_control = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load team_ball_control.pkl: {e}")
            team_ball_control = None
    else:
        player_assigner = PlayerBallAssigner()
        team_ball_control = []
        for frame_number, player_track in enumerate(tracks["players"]):
            ball_bbox = tracks["ball"][frame_number][1]["bbox"]
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
            if assigned_player != -1:
                tracks["players"][frame_number][assigned_player]["has_control"] = True

                team_ball_control.append(tracks["players"][frame_number][assigned_player]["team"])
            else:
                try:
                    team_ball_control.append(team_ball_control[-1])
                except Exception:
                    team_ball_control.append(1)
        team_ball_control = np.array(team_ball_control)
        # Security: Save outputs securely
        try:
            tracker.release()
            save_pkl("tracks.pkl", tracks)
            save_pkl("team_ball_control.pkl", team_ball_control)
            save_pkl("camera_movement_frames.pkl", camera_movement_frames)
        except Exception as e:
            print(f"[ERROR] Failed to save output files: {e}")


    option_frames = {
        "circle": [],
        "voronoi": [],
        "line": []
    }
    try:
        output_video_frames = tracker.draw_annotation(frames, tracks, team_ball_control, option_frames)
        output_video_frames = camera_movement.draw_camera_movement(output_video_frames, camera_movement_frames)
        output_video_frames = speed_and_distance.draw_speed_and_distance(output_video_frames, tracks)
    except Exception as e:
        print(f"[ERROR] Failed to annotate frames: {e}")
        sys.exit(1)

    # Output paths (use env var for outputs_dir)
    output_path = video_path.replace("inputs", os.path.basename(outputs_dir)).replace(".mp4", ".mp4")
    circle_path = output_path.replace(".mp4", "_circle.mp4")
    voronoi_path = output_path.replace(".mp4", "_voronoi.mp4")
    line_path = output_path.replace(".mp4", "_line.mp4")
    path_backend = [output_path, circle_path, voronoi_path, line_path]

    try:
        if output_video_frames:
            write_video(output_path, output_video_frames)
        else:
            print(f"[WARNING] No main output frames to write for {output_path}")
        if option_frames["circle"]:
            write_video(circle_path, option_frames["circle"])
        else:
            print(f"[WARNING] No circle frames to write for {circle_path}")
        if option_frames["voronoi"]:
            write_video(voronoi_path, option_frames["voronoi"])
        else:
            print(f"[WARNING] No voronoi frames to write for {voronoi_path}")
        if option_frames["line"]:
            write_video(line_path, option_frames["line"])
        else:
            print(f"[WARNING] No line frames to write for {line_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write output videos: {e}")
        sys.exit(1)
    return path_backend, tracks

if __name__ == "__main__":
    # Security: Get video path from env or argument, not hardcoded
    video_path = os.environ.get("VIDEO_PATH")
    if not video_path and len(sys.argv) > 1:
        video_path = sys.argv[1]
    if not video_path:
        print("[ERROR] Please provide a video path via the VIDEO_PATH environment variable or as a command-line argument.")
        sys.exit(1)

    # Set load_pkl from env or default to True
    load_pkl = os.environ.get("LOAD_PKL", "True").lower() == "true"

    # Security: Run process with validated input
    try:
        output_path, tracks = process(video_path, load_pkl=load_pkl)
        print(f"[INFO] Processing complete. Outputs: {output_path}")
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        sys.exit(1)