import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ButtonGradient from "./assets/svg/ButtonGradient"
import Header from './components/Header'
import Footer from './components/Footer'
import Hero from './components/Hero'
import Login from './components/Login'
import Signup from './components/Signup' 
import Benefits from './components/Benefits'
import Live from './components/Live'
const App = () => {

  return (
    <>
    <div className="pt-[4.75rem] lg:pt-[5.25rem] overflow-hidden">
        <Header />
        <Routes>
          <Route path="/" element={<Hero />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/live" element={<Live />} />
        </Routes>
        <Footer />
      </div>
      <ButtonGradient />
    </>
  )
}

export default App
