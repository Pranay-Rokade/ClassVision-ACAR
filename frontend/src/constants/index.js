
import {
    benefitIcon1,
    benefitIcon2,
    benefitIcon3,
    benefitIcon4,
    benefitImage2,
    homeSmile, 
    file02, 
    searchMd, 
    plusSquare,
    notification4,
    notification3,
    notification2,
    figma,
    notion,
    discord,
    slack,
    photoshop,
    protopie,
    framer,
    raindrop,
    cvacar,
    discordBlack,
    instagram,
    telegram,
  } from "../assets";
  
  export const navigation = [
    {
      id: "0",
      title: "Features",
      url: "/#features",
    },
    {
      id: "1",
      title: "How to use",
      url: "/#how-to-use",
    },
    {
      id: "2",
      title: "New account",
      url: "/signup",
      onlyMobile: true,
    },
    {
      id: "3",
      title: "Sign in",
      url: "/login",
      onlyMobile: true,
    },
  ];

  export const heroIcons = [homeSmile, file02, searchMd, plusSquare];
  export const notificationImages = [notification4, notification3, notification2];
  export const companyLogos = [cvacar, cvacar, cvacar, cvacar, cvacar];

  export const benefits = [
    {
      id: "0",
      title: "Fast Responding",
      text: "Deliver quick, relevant answers using intelligent automation, reducing response time and boosting productivity.",
      backgroundUrl: "./src/assets/benefits/card-1.svg",
      iconUrl: benefitIcon1,
      imageUrl: benefitImage2,
    },
    {
      id: "1",
      title: "Real-Time Activity Detection",
      text: "Instantly identifies and classifies student activities to keep teachers informed.",
      backgroundUrl: "./src/assets/benefits/card-2.svg",
      iconUrl: benefitIcon2,
      imageUrl: benefitImage2,
      light: true,
    },
    {
      id: "2",
      title: "Anomaly Detection",
      text: "Alerts teachers to unusual or inattentive behavior for better classroom management.",
      backgroundUrl: "./src/assets/benefits/card-3.svg",
      iconUrl: benefitIcon3,
      imageUrl: benefitImage2,
    },
    {
      id: "3",
      title: "Secure & Private Communication",
      text: "Implements top-tier encryption and privacy safeguards to protect user data and maintain confidentiality.",
      backgroundUrl: "./src/assets/benefits/card-4.svg",
      iconUrl: benefitIcon4,
      imageUrl: benefitImage2,
      light: true,
    },
    {
      id: "4",
      title: "Seamless Integration",
      text: "Easily integrates with existing classroom cameras and learning management systems.",
      backgroundUrl: "./src/assets/benefits/card-5.svg",
      iconUrl: benefitIcon1,
      imageUrl: benefitImage2,
    },
    {
      id: "5",
      title: "High-Performance Recognition",
      text: "Our system delivers accurate and reliable classroom activity detection, ensuring precise monitoring and insights for teachers.",
      backgroundUrl: "./src/assets/benefits/card-6.svg",
      iconUrl: benefitIcon2,
      imageUrl: benefitImage2,
    },
  ];

  export const collabText =
  "With smart automation and high performance, ClassVision is the perfect solution for teachers to monitor classrooms effortlessly.";

export const collabContent = [
  {
    id: "0",
    title: "Upload Video",
    text: collabText,
  },
  {
    id: "1",
    title: "Live Monitoring",
  },
  {
    id: "2",
    title: "Top-notch Security",
  },
];

  export const collabApps = [
    {
      id: "0",
      title: "Figma",
      icon: figma,
      width: 26,
      height: 36,
    },
    {
      id: "1",
      title: "Notion",
      icon: notion,
      width: 34,
      height: 36,
    },
    {
      id: "2",
      title: "Discord",
      icon: discord,
      width: 36,
      height: 28,
    },
    {
      id: "3",
      title: "Slack",
      icon: slack,
      width: 34,
      height: 35,
    },
    {
      id: "4",
      title: "Photoshop",
      icon: photoshop,
      width: 34,
      height: 34,
    },
    {
      id: "5",
      title: "Protopie",
      icon: protopie,
      width: 34,
      height: 34,
    },
    {
      id: "6",
      title: "Framer",
      icon: framer,
      width: 26,
      height: 34,
    },
    {
      id: "7",
      title: "Raindrop",
      icon: raindrop,
      width: 38,
      height: 32,
    },
  ];
  

  export const socials = [
    {
      id: "0",
      title: "Discord",
      iconUrl: discordBlack,
      url: "#",
    },
    {
      id: "1",
      title: "Instagram",
      iconUrl: instagram,
      url: "#",
    },
    {
      id: "2",
      title: "Telegram",
      iconUrl: telegram,
      url: "#",
    }
  ];
  