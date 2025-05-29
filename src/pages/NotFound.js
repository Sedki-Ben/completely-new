import React from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

function NotFound() {
  const { t } = useTranslation();

  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center text-center px-4">
      <h1 className="text-6xl font-bold text-slate-700 dark:text-slate-500 mb-4">404</h1>
      <h2 className="text-2xl font-serif mb-4 text-gray-900 dark:text-white">
        {t('Page Not Found')}
      </h2>
      <p className="text-gray-600 dark:text-gray-400 mb-8 max-w-md">
        {t('The page you are looking for might have been removed, had its name changed, or is temporarily unavailable.')}
      </p>
      <Link
        to="/"
        className="px-6 py-3 bg-slate-700 hover:bg-slate-800 text-white rounded-lg transition-colors duration-300"
      >
        {t('Back to Home')}
      </Link>
    </div>
  );
}

export default NotFound; 