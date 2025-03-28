import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ButtonGradient from "./assets/svg/ButtonGradient"
import Header from './components/Header'
import Footer from './components/Footer'
import Hero from './components/Hero'
import Login from './components/Login'
import Signup from './components/Signup' 
import Live from './components/Live'
<<<<<<< HEAD
import Upload from './components/Upload';
=======
import VideoStream from './components/Try';
>>>>>>> d2788a034112bb3a8f2abca0cc1b95cc53ccf3ed
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
<<<<<<< HEAD
          <Route path="/upload" element={<Upload/>} />
=======
          <Route path="/stream" element={<VideoStream />} />
>>>>>>> d2788a034112bb3a8f2abca0cc1b95cc53ccf3ed
        </Routes>
        <Footer />
      </div>
      <ButtonGradient />
    </>
  )
}

export default App
