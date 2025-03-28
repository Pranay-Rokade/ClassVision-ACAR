from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response


# Create your views here.

from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
import json

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
