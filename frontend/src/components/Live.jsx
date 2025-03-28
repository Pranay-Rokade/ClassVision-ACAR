import React, { useState, useRef } from "react";
import Button from "./Button"; // Assuming you have a Button component
import axios from "axios";

const LiveMonitoring = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isMonitoring, setIsMonitoring] = useState(false);  
  const videoRef = useRef(null);
  let mediaSource;

  // Backend URL for the live video stream
  const backendUrl = "http://127.0.0.1:8000/classify/livevideo";

  // Start Video Streaming on Button Click
  const handleStartVideo = () => {
    if (!isStreaming) {
      startStreaming();
      setIsStreaming(true);
    }
  };

  // Stop Video Streaming
  const handleStopVideo = () => {
    if (mediaSource) {
      mediaSource.endOfStream();
    }
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.src = "";
    }
    setIsStreaming(false);
  };

  // Function to Start Streaming
  const startStreaming = () => {
    if (videoRef.current) {
      videoRef.current.src = backendUrl; // Use backend URL as video source
      videoRef.current.play(); // Play video once loaded
    }
  };


  const toggleMonitoring = () => {
    if (isMonitoring) {
      handleStopVideo();
      setIsMonitoring(!isMonitoring);
    }
    else {
      handleStartVideo();
      setIsMonitoring(!isMonitoring);
    }
  };

  return (
    <div className="flex justify-center items-center w-full bg-gray-950 pt-[4.75rem] lg:pt-[5.25rem] px-4 mt-[-40px]">
      <div className="w-full max-w-[1600px]">
        {" "}
        {/* Increased max-width */}
        <div className="w-full bg-gradient-to-r from-blue-500 to-purple-600 rounded-3xl shadow-2xl p-2">
          <div className="border-8 border-transparent rounded-2xl bg-gray-900 shadow-2xl p-6">
            <div className="flex gap-6 h-[475px]">
              {/* Video Feed Section - 80% width */}
              <div className="w-[80%]">
                <h2 className="text-3xl font-bold text-gray-300 mb-4">
                  Live Monitoring Feed
                </h2>
                <div className="w-full aspect-video bg-gray-800 rounded-xl flex items-center justify-center h-[430px] overflow-auto">
                  {isMonitoring ? (
                    <div className="text-center w-full h-full">
                      {/* Laptop Mockup */}
                      {/* Video Frame Display */}
                      {/* {console.log(live_stream_url)} */}
                      <video src={"http://127.0.0.1:8000/classify/livevideo"} alt="Live Video"
                        className="w-full h-full border-2  rounded-lg border-gray-300 shadow-lg overflow-hidden " />
                      {/* <div className="w-full h-full bg-gray-700 rounded-lg overflow-hidden shadow-lg flex items-center justify-center">
                        <p className="text-2xl text-gray-300">
                          Active Monitoring in Progress
                        </p>
                      </div> */}
                    </div>
                  ) : (
                    <p className="text-2xl text-gray-500">
                      Monitoring Not Started
                    </p>
                  )}
                </div>
              </div>

              {/* Activity Log Section - 20% width */}
              <div className="w-[20%]">
                <h2 className="text-3xl font-bold text-gray-300 mb-4">
                  Activity
                </h2>
                <div className="bg-gray-800 rounded-xl p-4 h-[430px] overflow-auto">
                  {isMonitoring ? (
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-gray-700">
                          <th className="py-2 text-left text-gray-400">Time</th>
                          <th className="py-2 text-left text-gray-400">
                            Activity
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {[
                          {
                            timestamp: "10:15:23",
                            activity: "Paying attention",
                          },
                          { timestamp: "10:15:45", activity: "Taking notes" },
                          {
                            timestamp: "10:16:02",
                            activity: "Looking at phone",
                          },
                          {
                            timestamp: "10:16:30",
                            activity: "Listening to lecture",
                          },
                          { timestamp: "10:17:15", activity: "Writing" },
                          { timestamp: "10:17:45", activity: "Raised hand" },
                          {
                            timestamp: "10:18:10",
                            activity: "Collaborative work",
                          },
                          {
                            timestamp: "10:18:35",
                            activity: "Reading textbook",
                          },
                        ].map((activity, index) => (
                          <tr
                            key={index}
                            className="border-b border-gray-700 last:border-b-0 hover:bg-gray-700 transition-colors"
                          >
                            <td className="py-2 text-gray-300 text-sm">
                              {activity.timestamp}
                            </td>
                            <td className="py-2 text-gray-300 text-sm">
                              {activity.activity}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <p className="text-center text-gray-500">
                      No activities to display
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Control Button */}
            <div className="mt-6 flex justify-center">
              <Button
                onClick={toggleMonitoring}
                className="w-full max-w-md py-3 text-lg"
              >
                {isMonitoring ? "Stop Monitoring" : "Start Monitoring"}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveMonitoring;
