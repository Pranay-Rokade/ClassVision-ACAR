from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import StreamingHttpResponse, FileResponse, JsonResponse, HttpResponse
from django.core.files.storage import default_storage
from rest_framework.parsers import MultiPartParser, FormParser
import cv2
import os
from django.conf import settings


# Create your views here.
VIDEO_URL = None  # Placeholder for the video URL


# Receive Video URL Class
class receive_video_url(APIView):
    def post(self, request):  
        try:
            # Get the video URL from the request
            video_url = request.data.get("videourl")
            # video_url = "https://192.0.0.4:8080/video"

            if not video_url:
                return Response({"error": "No video URL provided"}, status=400)

            # Validate URL
            if not video_url.startswith("http"):
                return Response({"error": "Invalid video URL"}, status=400)

            # Stream the video for real-time processing
            return StreamingHttpResponse(
                generate_frames(video_url),
                content_type="multipart/x-mixed-replace;boundary=frame",
                status=200
            )

        except Exception as e:
            return Response({"error": str(e)}, status=500)
        
    def get(self, request):  
        try:
            # Get the video URL from the request
            # video_url = request.data.get("videourl")
            video_url = "https://192.0.0.4:8080/video"

            if not video_url:
                return Response({"error": "No video URL provided"}, status=400)

            # Validate URL
            if not video_url.startswith("http"):
                return Response({"error": "Invalid video URL"}, status=400)

            # Stream the video for real-time processing
            return StreamingHttpResponse(
                generate_frames(video_url),
                content_type="multipart/x-mixed-replace;boundary=frame",
                status=200
            )

        except Exception as e:
            return Response({"error": str(e)}, status=500)


# Video frame generator for live video processing
def generate_frames(video_url):
    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        yield b""

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Process frame (Add your custom analysis logic here)
        frame = process_frame(frame)

        # Encode and yield the frame
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()


# Example function to process the frame
def process_frame(frame):
    # Add any frame processing logic here (e.g., object detection, pose estimation, etc.)
    cv2.putText(frame, "Processing...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


class VideoClassification(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        try:
            # Get video file from request
            video_file = request.FILES.get("video")

            if not video_file:
                return Response({"error": "No video file provided"}, status=400)

            # Save the uploaded video temporarily
            video_path = default_storage.save("temp/" + video_file.name, video_file)
            video_full_path = os.path.join(default_storage.location, video_path)

            # Process the video
            processed_video_path = process_video(video_full_path)

            # Send the processed video back to React
            response = FileResponse(open(processed_video_path, "rb"), content_type="video/mp4", status=200)
            response["Content-Disposition"] = f'attachment; filename="processed_video.mp4"'

            # Clean up temporary files
            os.remove(video_full_path)
            os.remove(processed_video_path)

            return response

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


def process_video(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Define output path for processed video
    output_path = os.path.join("media/processed_videos", "processed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add custom text or processing here
        cv2.putText(frame, "Processed Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write processed frame to new video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()

    return output_path
