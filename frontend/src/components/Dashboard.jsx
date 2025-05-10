import React from "react";
import "../dashboard.css";
import {
  BsFillArchiveFill,
  BsFillGrid3X3GapFill,
  BsPeopleFill,
  BsFillBellFill,
} from "react-icons/bs";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from "recharts";
import { useEffect, useState } from "react";
import axios from "axios";

const Dashboard = () => {
  const [donoughtData, setDonoughtData] = useState([]);
  const [barData, setBarData] = useState([]);
  const [pieData, setPieData] = useState([]);
  const [lineData, setLineData] = useState([]);

  const [sudents, setStudents] = useState();
  const [activities, setActivities] = useState();
  const [positiveActivities, setPositiveActivities] = useState();
  const [negativeActivities, setNegativeActivities] = useState();


  useEffect(() => {
    let bardata = [];
    let piedata = [];
    let linedata = [];
    axios
      .get("http://127.0.0.1:8000/analysis/positive-negative-stats")
      .then((response) => {
        const res = response.data;
        setDonoughtData([
          { name: "Positive", value: res.positive },
          { name: "Negative", value: res.negative },
        ]);

        axios
          .get("http://127.0.0.1:8000/analysis/activity-count")
          .then((response) => {
            const res = response.data;
            bardata = Object.entries(res).map(([key, value]) => ({
              name: key,
              value: value,
            }));
            setBarData(bardata);
          });

        axios
          .get("http://127.0.0.1:8000/analysis/percentage-of-actions")
          .then((response) => {
            const res = response.data;
            piedata = Object.entries(res).map(([key, value]) => ({
              name: key,
              value: value,
            }));
            setPieData(piedata);
          });

        axios
          .get("http://127.0.0.1:8000/analysis/activities-per-student")
          .then((response) => {
            const res = response.data;
            linedata = Object.entries(res).map(([key, value]) => ({
              name: key,
              value: value,
            }));
            setLineData(linedata);
          });

        axios
          .get("http://127.0.0.1:8000/analysis/kpis")
          .then((response) => {
            const res = response.data;
            setStudents(res.total_students);
            setActivities(res.total_unique_activities);  
            setPositiveActivities(res.positive_activities);
            setNegativeActivities(res.negative_activities);
          });

        
      })

      .catch((error) => {
        console.error("Error fetching activity stats:", error);
      });
  }, []);
  // Updated data to match the image

  const COLORS = ["#514ae0", "#00C49F", "#FFBB28", "#FF8042", "#FF6FCF", "#36A2EB"]  ;

  const COLORS2 = ["#00C49F", "#FF8042"]; // green = positive, orange = negative

  return (
    <main className="dashboard-container">
      {/* <div className="dashboard-title">
        <h3>DASHBOARD</h3>
      </div> */}

      <div className="dashboard-cards">
        <div className="card" style={{ backgroundColor: "#2962ff" }}>
          <div className="card-inner">
            <h3>STUDENTS</h3>
            <BsFillArchiveFill className="card_icon" />
          </div>
          <h1>{sudents}</h1>
        </div>
        <div className="card" style={{ backgroundColor: "#ff6d00" }}>
          <div className="card-inner">
            <h3>ACTIVITES PERFORMED</h3>
            <BsFillGrid3X3GapFill className="card_icon" />
          </div>
          <h1>{activities}</h1>
        </div>
        <div className="card" style={{ backgroundColor: "#2e7d32" }}>
          <div className="card-inner">
            <h3>POSITIVE ACTIVITES PERFORMED</h3>
            <BsPeopleFill className="card_icon" />
          </div>
          <h1>{positiveActivities}</h1>
        </div>
        <div className="card" style={{ backgroundColor: "#d50000" }}>
          <div className="card-inner">
            <h3>NEGATIVE ACTIVITIES PERFORMED</h3>
            <BsFillBellFill className="card_icon" />
          </div>
          <h1>{negativeActivities}</h1>
        </div>
      </div>

      <div className="charts">
        <div className="chart-container">
          <h3>Number of Times Activity Detected</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 10]} />
              <Tooltip />
              <Legend />
              <Bar dataKey="value" fill="#514ae0" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <h3>Suspect Performance</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={lineData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 10]} />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#514ae0"
                activeDot={{ r: 8 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bottom-charts">
        <div className="chart-container">
          <h3>Teacher Attention Required Distribution</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }) =>
                  `${name} ${(percent * 100).toFixed(0)}%`
                }
              >
                {pieData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <h3 className="text-xl font-semibold mb-2">Productivity Overview</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={donoughtData}
                cx="50%"
                cy="50%"
                innerRadius={70}
                outerRadius={100}
                fill="#8884d8"
                paddingAngle={5}
                dataKey="value"
                label
              >
                {donoughtData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS2[index % COLORS2.length]}
                  />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </main>
  );
};

export default Dashboard;
