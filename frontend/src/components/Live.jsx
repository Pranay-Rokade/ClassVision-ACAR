import React, { useState } from 'react';

const LiveMonitoring = () => {
  const [isMonitoring, setIsMonitoring] = useState(false);

  const toggleMonitoring = () => {
    setIsMonitoring(!isMonitoring);
  };

  return (
    <div className="flex justify-center items-center min-h-screen w-full bg-gray-950 pt-[4.75rem] lg:pt-[5.25rem] px-4">
      <div className="w-full max-w-[1600px]"> {/* Increased max-width */}
        <div className="w-full bg-gradient-to-r from-blue-500 to-purple-600 rounded-3xl shadow-2xl p-2">
          <div className="border-8 border-transparent rounded-2xl bg-gray-900 shadow-2xl p-6">
            <div className="flex gap-6">
              {/* Video Feed Section - 80% width */}
              <div className="w-[80%]">
                <h2 className="text-3xl font-bold text-gray-300 mb-4">
                  Live Monitoring Feed
                </h2>
                <div className="w-full aspect-video bg-gray-800 rounded-xl flex items-center justify-center">
                  {isMonitoring ? (
                    <div className="text-center w-full h-full">
                      {/* Laptop Mockup */}
                      <div className="w-full h-full bg-gray-700 rounded-lg overflow-hidden shadow-lg flex items-center justify-center">
                        <p className="text-2xl text-gray-300">
                          Active Monitoring in Progress
                        </p>
                      </div>
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
                  Activity Log
                </h2>
                <div className="bg-gray-800 rounded-xl p-4 h-[600px] overflow-auto">
                  {isMonitoring ? (
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-gray-700">
                          <th className="py-2 text-left text-gray-400">Time</th>
                          <th className="py-2 text-left text-gray-400">Activity</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[
                          { timestamp: '10:15:23', activity: 'Paying attention' },
                          { timestamp: '10:15:45', activity: 'Taking notes' },
                          { timestamp: '10:16:02', activity: 'Looking at phone' },
                          { timestamp: '10:16:30', activity: 'Listening to lecture' },
                          { timestamp: '10:17:15', activity: 'Writing' },
                          { timestamp: '10:17:45', activity: 'Raised hand' },
                          { timestamp: '10:18:10', activity: 'Collaborative work' },
                          { timestamp: '10:18:35', activity: 'Reading textbook' },
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
              <button 
                onClick={toggleMonitoring} 
                className="w-full max-w-md py-3 text-lg 
                  bg-gradient-to-r from-blue-500 to-purple-600 
                  text-white rounded-xl 
                  hover:from-blue-600 hover:to-purple-700 
                  transition-colors duration-300 
                  focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
              >
                {isMonitoring ? 'Stop Monitoring' : 'Start Monitoring'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveMonitoring;