import React, { createContext, useState, useContext, useEffect } from 'react';
import i18next from 'i18next';

const LanguageContext = createContext();

export const LanguageProvider = ({ children }) => {
  const [language, setLanguage] = useState(i18next.language || 'en');
  const [dir, setDir] = useState(i18next.language === 'ar' ? 'rtl' : 'ltr');

  useEffect(() => {
    // Update document direction based on language
    document.documentElement.dir = dir;
    document.documentElement.lang = language;
  }, [dir, language]);

  useEffect(() => {
    // Listen for i18next language changes
    const handleLangChange = (lng) => {
      setLanguage(lng);
      setDir(lng === 'ar' ? 'rtl' : 'ltr');
    };
    i18next.on('languageChanged', handleLangChange);
    return () => {
      i18next.off('languageChanged', handleLangChange);
    };
  }, []);

  const changeLanguage = async (lng) => {
    try {
      await i18next.changeLanguage(lng);
      // setLanguage and setDir will be handled by the event listener
      localStorage.setItem('i18nextLng', lng);
      // Update user preference if logged in
      const token = localStorage.getItem('token');
      if (token) {
        await fetch('/api/users/language', {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify({ language: lng })
        });
      }
    } catch (error) {
      console.error('Failed to change language:', error);
    }
  };

  return (
    <LanguageContext.Provider value={{ language, changeLanguage, dir }}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
}; 
 