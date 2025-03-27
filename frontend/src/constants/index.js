
import {
    homeSmile, 
    file02, 
    searchMd, 
    plusSquare,
    notification4,
    notification3,
    notification2,
    yourlogo,
    discordBlack,
    instagram,
    telegram,
  } from "../assets";
  
  export const navigation = [
    {
      id: "0",
      title: "Features",
      url: "#features",
    },
    {
      id: "1",
      title: "How to use",
      url: "#how-to-use",
    },
    {
      id: "2",
      title: "New account",
      url: "#signup",
      onlyMobile: true,
    },
    {
      id: "3",
      title: "Sign in",
      url: "#login",
      onlyMobile: true,
    },
  ];

  export const heroIcons = [homeSmile, file02, searchMd, plusSquare];
  export const notificationImages = [notification4, notification3, notification2];
  export const companyLogos = [yourlogo, yourlogo, yourlogo, yourlogo, yourlogo];
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
  