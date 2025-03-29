import React, { useState, useRef } from 'react';
import axios from 'axios';
import Button from './Button';

const Upload2 = () => {
    const [originalVideoFile, setOriginalVideoFile] = useState(null);
    const [processedVideoUrl, setProcessedVideoUrl] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [errorMessage, setErrorMessage] = useState(null);
    const fileInputRef = useRef(null);
    const videoRef = useRef(null);
  
    const handleFileChange = (event) => {
      const file = event.target.files[0];
      if (file && file.type.startsWith('video/')) {
        setOriginalVideoFile(file);
        setErrorMessage(null);
      } else {
        alert('Please upload a valid video file.');
      }
    };

    const handleVideoUpload = async () => {
      if (!originalVideoFile) {
        setErrorMessage('Please select a video file first.');
        return;
      }

      // Create FormData to send the file
      const formData = new FormData();
      formData.append('video', originalVideoFile);

      try {
        setIsLoading(true);
        setErrorMessage(null);
        
        // Revoke any existing object URL to prevent memory leaks
        if (processedVideoUrl) {
          URL.revokeObjectURL(processedVideoUrl);
        }
        
        // Use Axios to send the request to your Django backend
        const response = await axios.post('http://127.0.0.1:8000/classify/videoclassification', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          responseType: 'blob', // Important for handling file download
        });

        // Check if response is valid
        if (!response.data || response.data.size === 0) {
          throw new Error('Received empty response from server');
        }

        // Check the content type of the response
        const contentType = response.headers['content-type'];
        console.log('Response content type:', contentType);
        
        // If it's an error message as JSON, try to parse it
        if (contentType && contentType.includes('application/json')) {
          const reader = new FileReader();
          reader.onload = () => {
            try {
              const jsonResponse = JSON.parse(reader.result);
              setErrorMessage(jsonResponse.message || 'Server returned an error');
            } catch (e) {
              setErrorMessage('Invalid response from server');
            }
          };
          reader.readAsText(response.data);
          return;
        }
        
        // Ensure we have a video MIME type
        const blobType = contentType && contentType.startsWith('video/') 
          ? contentType 
          : 'video/mp4';
        
        // Create a blob URL for the processed video
        const processedBlob = new Blob([response.data], { type: blobType });
        const processedVideoBlob = URL.createObjectURL(processedBlob);
        
        setProcessedVideoUrl(processedVideoBlob);
        
        // Additional logging for debugging
        console.log('Processed Video Blob Size:', processedBlob.size);
        console.log('Processed Video MIME Type:', processedBlob.type);
        console.log('Processed Video URL:', processedVideoBlob);
        
        // Show success alert
        alert('Video processed successfully!');
      } catch (error) {
        console.error('Error processing video:', error);
        
        // More detailed error handling
        let errorMsg = 'Processing failed';
        if (error.response) {
          // Try to parse error response if it's not a blob
          if (error.response.data instanceof Blob) {
            const reader = new FileReader();
            reader.onload = () => {
              try {
                const text = reader.result;
                setErrorMessage(`Server Error: ${text}`);
              } catch {
                setErrorMessage(`Server Error: ${error.response.statusText}`);
              }
            };
            reader.readAsText(error.response.data);
            return;
          } else {
            errorMsg = `Server Error: ${error.response.statusText}`;
          }
        } else if (error.request) {
          errorMsg = 'No response received from server';
        } else {
          errorMsg = error.message || 'Unknown error occurred';
        }
        
        setErrorMessage(errorMsg);
      } finally {
        setIsLoading(false);
      }
    };

    const handleButtonClick = () => {
      fileInputRef.current.value = ''; // Reset input to allow selecting same file again
      fileInputRef.current.click();
    };

    const handleReset = () => {
      setOriginalVideoFile(null);
      
      // Revoke and clear processed video URL
      if (processedVideoUrl) {
        URL.revokeObjectURL(processedVideoUrl);
        setProcessedVideoUrl(null);
      }
      
      setErrorMessage(null);
    };
  
    return (
      <div className="flex justify-center items-center w-full bg-gray-950 pt-[4.75rem] lg:pt-[5.25rem] px-4 mt-[-40px] pb-10">
        <div className="w-full max-w-[800px]">
          <div className="w-full bg-gradient-to-r from-blue-500 to-purple-600 rounded-3xl shadow-2xl p-2">
            <div className="border-8 border-transparent rounded-2xl bg-gray-900 shadow-2xl p-6">
              <h2 className="text-3xl font-bold text-gray-300 mb-4 text-center">
                Video Processing
              </h2>
              <div className="w-full bg-gray-800 rounded-xl p-6 flex flex-col items-center">
                {/* Hidden file input */}
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleFileChange}
                  ref={fileInputRef}
                  className="hidden"
                />
                
                {/* Video Preview Area */}
                <div className="w-full mb-4">
                  {processedVideoUrl ? (
                    <div className="w-full relative">
                      <video 
                        ref={videoRef}
                        src={processedVideoUrl}
                        controls 
                        className="w-full rounded-lg shadow-lg"
                        onError={(e) => {
                          console.error('Video error:', e);
                          setErrorMessage('Failed to load video. The response may not be a valid video file.');
                        }}
                      >
                        Your browser does not support the video tag.
                      </video>
                      <div className="absolute top-2 right-2">
                        <button
                          onClick={handleReset}
                          className="bg-red-500 hover:bg-red-600 text-white p-2 rounded-full shadow-lg"
                          title="Remove video"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  ) : originalVideoFile ? (
                    <div className="w-full h-52 bg-gray-700 rounded-lg flex flex-col items-center justify-center text-gray-500">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                      <span>{originalVideoFile.name} Selected</span>
                    </div>
                  ) : (
                    <div className="w-full h-52 bg-gray-700 rounded-lg flex flex-col items-center justify-center text-gray-500">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                      <span>No video uploaded</span>
                    </div>
                  )}
                </div>
                
                {/* Action Buttons */}
                <div className="flex gap-4 mt-4">
                  <Button 
                    onClick={handleButtonClick}
                  >
                    {originalVideoFile ? 'Change File' : 'Choose File'}
                  </Button>
                  
                  <Button 
                    onClick={handleVideoUpload}
                    disabled={!originalVideoFile || isLoading}
                  >
                    {isLoading ? 'Processing...' : 'Process Video'}
                  </Button>
                </div>

                {/* Error Message Display */}
                {errorMessage && (
                  <div className="mt-4 text-red-500 bg-red-100 p-3 rounded-lg">
                    {errorMessage}
                  </div>
                )}

                {/* Download button for processed video */}
                {processedVideoUrl && !errorMessage && (
                  <div className="mt-4">
                    <a 
                      href={processedVideoUrl} 
                      download="processed_video.mp4"
                      className="bg-green-500 hover:bg-green-600 text-white py-2 px-4 rounded-lg"
                    >
                      Download Processed Video
                    </a>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };
  
  export default Upload2;