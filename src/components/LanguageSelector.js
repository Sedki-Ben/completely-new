import React from 'react';
import { useTranslation } from 'react-i18next';
import { FiGlobe } from 'react-icons/fi';

const languages = [
  { code: 'en', name: 'English', dir: 'ltr' },
  { code: 'fr', name: 'Français', dir: 'ltr' },
  { code: 'ar', name: 'العربية', dir: 'rtl' }
];

function LanguageSelector() {
  const { i18n } = useTranslation();

  const changeLanguage = (lng) => {
    i18n.changeLanguage(lng);
    // Update document direction for RTL support
    document.documentElement.dir = languages.find(lang => lang.code === lng)?.dir || 'ltr';
  };

  return (
    <div className="relative group" dir="ltr">
      <button
        className="p-2 rounded-lg text-gray-600 dark:text-gray-300 hover:bg-slate-50 dark:hover:bg-gray-700 transition-all duration-300"
        aria-label="Select language"
      >
        <FiGlobe className="w-5 h-5" />
      </button>
      <div className="absolute right-0 mt-2 py-2 w-48 bg-white dark:bg-gray-800 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-300 z-50">
        {languages.map((language) => (
          <button
            key={language.code}
            className={`w-full text-left px-4 py-2 text-sm hover:bg-slate-50 dark:hover:bg-gray-700 transition-colors ${
              i18n.language === language.code
                ? 'text-slate-700 dark:text-slate-400 font-medium'
                : 'text-gray-700 dark:text-gray-300'
            }`}
            onClick={() => changeLanguage(language.code)}
          >
            {language.name}
          </button>
        ))}
      </div>
    </div>
  );
}

export default LanguageSelector; 