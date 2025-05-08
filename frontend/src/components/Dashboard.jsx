import React from 'react';
import '../dashboard.css';
import {
  BsFillArchiveFill,
  BsFillGrid3X3GapFill,
  BsPeopleFill,
  BsFillBellFill
} from 'react-icons/bs';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
  LineChart, Line, PieChart, Pie, Cell, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';

const Dashboard = () => {
  // Updated data to match the image
  const barData = [
    { name: 'Mobile', value: 20 },
    { name: 'Eating', value: 10 },
    { name: 'HandRaise', value: 15 },
    { name: 'Sleeping', value: 2 },
    { name: 'Reading', value: 50 },
    { name: 'Writing', value: 30 },
    { name: 'Sitting', value: 70 },
  ];
  
  const pieData = [
    { name: 'Using Mobile', value: 25 },
    { name: 'Eating', value: 24 },
    { name: 'Hand Raise', value: 17 },
    { name: 'Sleeing', value: 34 },
  ];

  const radarData = [
    { subject: 'Hand Raise', score: 120, fullMark: 150 },
    { subject: 'Using Mobile', score: 85, fullMark: 150 },
    { subject: 'Reading', score: 130, fullMark: 150 },
    { subject: 'Eating', score: 90, fullMark: 150 },
    { subject: 'Writing', score: 130, fullMark: 150 },
    { subject: 'Sleeping', score: 100, fullMark: 150 },
  ];

  const COLORS = ['#514ae0', '#00C49F', '#FFBB28', '#FF8042'];

  return (
    <main className='dashboard-container'>
      <div className='dashboard-title'>
        <h3>DASHBOARD</h3>
      </div>

      <div className='dashboard-cards'>
        <div className='card' style={{ backgroundColor: '#2962ff' }}>
          <div className='card-inner'>
            <h3>STUDENTS</h3>
            <BsFillArchiveFill className='card_icon' />
          </div>
          <h1>3</h1>
        </div>
        <div className='card' style={{ backgroundColor: '#ff6d00' }}>
          <div className='card-inner'>
            <h3>ACTIVITES PERFORMED</h3>
            <BsFillGrid3X3GapFill className='card_icon' />
          </div>
          <h1>10</h1>
        </div>
        <div className='card' style={{ backgroundColor: '#2e7d32' }}>
          <div className='card-inner'>
            <h3>POSITIVE ACTIVITES PERFORMED</h3>
            <BsPeopleFill className='card_icon' />
          </div>
          <h1>8</h1>
        </div>
        <div className='card' style={{ backgroundColor: '#d50000' }}>
          <div className='card-inner'>
            <h3>NEGATIVE ACTIVITIES PERFORMED</h3>
            <BsFillBellFill className='card_icon' />
          </div>
          <h1>2</h1>
        </div>
      </div>

      <div className='charts'>
        <div className='chart-container'>
          <h3>Number of Times Activity Detected</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Legend />
              <Bar dataKey="value" fill="#514ae0" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className='chart-container'>
          <h3>Suspect Performance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="value" stroke="#514ae0" activeDot={{ r: 8 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className='bottom-charts'>
        <div className='chart-container'>
          <h3>Teacher Attention Required Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className='chart-container'>
          <h3>Student Performance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="subject" />
              <PolarRadiusAxis angle={30} domain={[0, 150]} />
              <Radar name="Student" dataKey="score" stroke="#514ae0" fill="#514ae0" fillOpacity={0.6} />
              <Tooltip />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </main>
  );
};

export default Dashboard;