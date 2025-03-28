import React, { useState } from "react";

const VideoStream = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [videoSrc, setVideoSrc] = useState("");

  // Backend URL for the live video stream
  const backendUrl = "http://127.0.0.1:8000/classify/livevideo/"; // Correct backend URL

  // Start Video Streaming on Button Click
  const handleStartVideo = () => {
    if (!isStreaming) {
      setVideoSrc(backendUrl); // Set backend video URL as src for <img>
      setIsStreaming(true);
    }
  };

  // Stop Video Streaming
  const handleStopVideo = () => {
    setVideoSrc(""); // Clear video source to stop streaming
    setIsStreaming(false);
  };

  return (
    <div style={{ textAlign: "center", marginTop: "20px" }}>
      <h2>📹 Live Video Feed</h2>

      {/* Start Live Video Button */}
      <button
        onClick={handleStartVideo}
        style={{
          padding: "10px 20px",
          backgroundColor: isStreaming ? "#ccc" : "#4CAF50",
          color: "#fff",
          border: "none",
          borderRadius: "5px",
          cursor: isStreaming ? "not-allowed" : "pointer",
          marginBottom: "20px",
          marginRight: "10px",
        }}
        disabled={isStreaming}
      >
        {isStreaming ? "Live Video Started" : "Start Live Video"}
      </button>

      {/* Stop Live Video Button */}
      <button
        onClick={handleStopVideo}
        style={{
          padding: "10px 20px",
          backgroundColor: "#FF4C4C",
          color: "#fff",
          border: "none",
          borderRadius: "5px",
          cursor: !isStreaming ? "not-allowed" : "pointer",
        }}
        disabled={!isStreaming}
      >
        Stop Live Video
      </button>

      {/* Display Live Video */}
      <div style={{ marginTop: "20px" }}>
        {videoSrc && (
          <img
            src={videoSrc}
            alt="Live Video"
            style={{
              width: "80%",
              height: "auto",
              border: "2px solid #ccc",
              borderRadius: "10px",
            }}
          />
        )}
      </div>
    </div>
  );
};

export default VideoStream;
