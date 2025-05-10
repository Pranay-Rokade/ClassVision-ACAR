import React, { useEffect,useState } from "react";
import { Link } from "react-router-dom";
import Button from "../components/Button";
import axios from "axios";

const InputField = ({ id, type, placeholder, value, onChange }) => {
    return (
      <div className="relative w-full">
        <input
          id={id}
          className="w-full bg-transparent text-gray-300 placeholder-gray-500 outline-none border-none 
                     px-4 py-2 text-lg relative z-10"
          type={type}
          placeholder={placeholder}
          value={value}
          onChange={onChange}
          required
        />
  
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox="0 0 400 40"
          preserveAspectRatio="none"
        >
          <defs>
            <linearGradient id="input-border-small" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#00FFFF" />
              <stop offset="50%" stopColor="#8000FF" />
              <stop offset="100%" stopColor="#FF00FF" />
            </linearGradient>
          </defs>
  
          <rect
            x="1" y="1" width="398" height="38" rx="10" ry="10"
            fill="none" stroke="url(#input-border-small)" strokeWidth="1"
          />
        </svg>
      </div>
    );
  };
  
const Signup = () => {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (password !== confirmPassword) {
      alert("Passwords do not match!");
      return;
    }
    axios.post
    console.log("Username:", username);
    console.log("Email:", email);
    console.log("Password:", password);
  };
  useEffect(() => {
    // Prevent scrolling when Login is mounted
    document.body.classList.add("overflow-hidden");
  
    // Cleanup: Re-enable scrolling on unmount
    return () => {
      document.body.classList.remove("overflow-hidden");
    };
  }, []);
  

  return (
    <div className="flex justify-center items-center mt-10 mb-5 w-screen">
      <div className="w-full max-w-lg">
        <section id="back-div" className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-3xl shadow-2xl p-2">
          <div className="border-8 border-transparent rounded-2xl bg-white dark:bg-gray-900 shadow-2xl p-6">
            <h1 className="text-4xl font-bold text-center cursor-default dark:text-gray-300 text-gray-900 mb-4">
              Sign Up
            </h1>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label htmlFor="username" className="block mb-1 text-lg dark:text-gray-300">
                  Username
                </label>
                <InputField
                  id="username"
                  type="text"
                  placeholder="Username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                />
              </div>
              <div>
                <label htmlFor="email" className="block mb-1 text-lg dark:text-gray-300">
                  Email
                </label>
                <InputField
                  id="email"
                  type="email"
                  placeholder="Email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
              <div>
                <label htmlFor="password" className="block mb-1 text-lg dark:text-gray-300">
                  Password
                </label>
                <InputField
                  id="password"
                  type="password"
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </div>
              <div>
                <label htmlFor="confirmPassword" className="block mb-1 text-lg dark:text-gray-300">
                  Confirm Password
                </label>
                <InputField
                  id="confirmPassword"
                  type="password"
                  placeholder="Confirm Password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                />
              </div>
              <Button className="w-full py-3 text-lg" type="submit">
                Sign Up
              </Button>
            </form>
            <div className="flex flex-col mt-4 text-sm text-center dark:text-gray-300">
              <p>
                Already have an account?
                <Link to="/login" className="text-blue-400 transition hover:underline">
                  Log in
                </Link>
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default Signup;