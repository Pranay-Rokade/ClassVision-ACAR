import React, { useState } from "react";
import { Link } from "react-router-dom";
import Button from "../components/Button";

import axios from "axios";

const InputField = ({ id, type, placeholder, value, onChange }) => {
    return (
      <div className="relative w-full">
        <input
          id={id}
          className="w-full bg-transparent text-gray-300 placeholder-gray-500 outline-none border-none 
                     px-6 py-4 text-xl relative z-10"
          type={type}
          placeholder={placeholder}
          value={value}
          onChange={onChange}
          required
        />
  
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox="0 0 500 50"
          preserveAspectRatio="none"
        >
          <defs>
            <linearGradient id="input-border-large" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#00FFFF" />
              <stop offset="50%" stopColor="#8000FF" />
              <stop offset="100%" stopColor="#FF00FF" />
            </linearGradient>
          </defs>
  
          <rect
            x="2" y="2" width="496" height="46" rx="12" ry="12"
            fill="none" stroke="url(#input-border-large)" strokeWidth="1"
          />
        </svg>
      </div>
    );
};

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    const data = {
      email: email,
      password: password,
    }

    try {
      const response = await axios.post("http://127.0.0.1:8000/auth/login", data, {
        headers: {
          "Content-Type": "application/json",
        },
      });
      console.log("Response:", response);
    } catch (error) {
      
    }

    console.log("Email:", email);
    console.log("Password:", password);
  };

  return (
    <div className="flex justify-center items-center h-screen w-full bg-gray-950 mt-[-60px]">
      <div className="w-full max-w-lg bg-gradient-to-r from-blue-500 to-purple-600 rounded-3xl shadow-2xl p-1">
        <div className="border-8 border-transparent rounded-2xl bg-white dark:bg-gray-900 shadow-2xl p-6">
          <h1 className="text-5xl font-bold text-center dark:text-gray-300 text-gray-900 mb-6">
            Log in
          </h1>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="email" className="block mb-2 text-lg dark:text-gray-300">
                Email
              </label>
              <InputField
                id="email"
                type="email"
                placeholder="Enter your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <div>
              <label htmlFor="password" className="block mb-2 text-lg dark:text-gray-300">
                Password
              </label>
              <InputField
                id="password"
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
            <Button className="w-full py-3 text-lg" type="submit">
              Login in
            </Button>
          </form>
          <div className="mt-4 text-center text-sm dark:text-gray-300">
            <p>
              Don't have an account? 
              <Link to="/signup" className="text-blue-400 transition hover:underline ml-1">
                Sign Up
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;