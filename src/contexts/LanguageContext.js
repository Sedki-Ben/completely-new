import React, { createContext, useState, useContext, useEffect } from 'react';
import i18next from 'i18next';
import { initReactI18next } from 'react-i18next';
import HttpApi from 'i18next-http-backend';

// Initialize i18next
i18next
  .use(HttpApi)
  .use(initReactI18next)
  .init({
    fallbackLng: 'en',
    supportedLngs: ['en', 'fr', 'ar'],
    ns: ['translation'],
    defaultNS: 'translation',
    backend: {
      loadPath: '/api/locales/{{lng}}/{{ns}}',
    },
    lng: 'en', // Force English as default
    load: 'languageOnly',
    interpolation: {
      escapeValue: false
    },
    react: {
      useSuspense: false
    },
    detection: {
      order: [], // Empty array to disable detection
      caches: [] // Disable caching
    }
  });

const LanguageContext = createContext();

export const LanguageProvider = ({ children }) => {
  const [language, setLanguage] = useState('en');
  const [dir, setDir] = useState('ltr');

  useEffect(() => {
    // Update document direction based on language
    document.documentElement.dir = dir;
    document.documentElement.lang = language;
  }, [dir, language]);

  const changeLanguage = async (lng) => {
    try {
      await i18next.changeLanguage(lng);
      setLanguage(lng);
      setDir(lng === 'ar' ? 'rtl' : 'ltr');
      
      // Store language preference
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
 