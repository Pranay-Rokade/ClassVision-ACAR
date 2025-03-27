import React from 'react'
import Button from './components/Button'
import ButtonGradient from "./assets/svg/ButtonGradient"
import Header from './components/Header'
import Footer from './components/Footer'
import Hero from './components/Hero'
const App = () => {

  return (
    <>
    <div className="pt-[4.75rem] lg:pt-[5.25rem] overflow-hidden">
      <Header />
      <Hero />
      <Button className="mt-10" href="#login">
        something
      </Button>
      <Footer />
    </div>
    <ButtonGradient />
    </>
  )
}

export default App
