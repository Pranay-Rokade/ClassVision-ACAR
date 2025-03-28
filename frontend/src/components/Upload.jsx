import React, { useState, useRef } from 'react';
import Button from './Button';

const Upload = () => {
    const [videoFile, setVideoFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const fileInputRef = useRef(null);
  
    const handleFileChange = (event) => {
      const file = event.target.files[0];
      if (file && file.type.startsWith('video/')) {
        setVideoFile(file);
        setPreviewUrl(URL.createObjectURL(file));
      } else {
        alert('Please upload a valid video file.');
      }
    };

    const handleButtonClick = () => {
      fileInputRef.current.value = ''; // Reset input to allow selecting same file again
      fileInputRef.current.click();
    };

    const handleRemoveVideo = () => {
      setVideoFile(null);
      setPreviewUrl(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = ''; // Clear the file input
      }
      // Revoke the object URL to avoid memory leaks
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  
    return (
      <div className="flex justify-center items-center w-full bg-gray-950 pt-[4.75rem] lg:pt-[5.25rem] px-4 mt-[-40px] pb-10">
        <div className="w-full max-w-[800px]">
          <div className="w-full bg-gradient-to-r from-blue-500 to-purple-600 rounded-3xl shadow-2xl p-2">
            <div className="border-8 border-transparent rounded-2xl bg-gray-900 shadow-2xl p-6">
              <h2 className="text-3xl font-bold text-gray-300 mb-4 text-center">
                Upload Video
              </h2>
              <div className="w-full bg-gray-800 rounded-xl p-6 flex flex-col items-center">
                {previewUrl ? (
                  <div className="w-full relative">
                    <video controls className="w-full rounded-lg shadow-lg">
                      <source src={previewUrl} type={videoFile.type} />
                      Your browser does not support the video tag.
                    </video>
                    <div className="absolute top-2 right-2">
                      <button
                        onClick={handleRemoveVideo}
                        className="bg-red-500 hover:bg-red-600 text-white p-2 rounded-full shadow-lg"
                        title="Remove video"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
                        </svg>
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="w-full h-52 bg-gray-700 rounded-lg flex flex-col items-center justify-center text-gray-500">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    <span>No video uploaded</span>
                  </div>
                )}
  
                {/* Hidden file input */}
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleFileChange}
                  ref={fileInputRef}
                  className="hidden"
                />
                
                <div className="flex gap-4 mt-4">
                  <Button 
                    onClick={handleButtonClick}
                  >
                    {previewUrl ? 'Change Video' : 'Choose File'}
                  </Button>
                  
                  {previewUrl && (
                    <Button 
                      onClick={handleRemoveVideo}
                    >
                      Remove Video
                    </Button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };
  
  export default Upload;