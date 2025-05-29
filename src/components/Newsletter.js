import React from 'react';
import { useTranslation } from 'react-i18next';
import { useLocation } from 'react-router-dom';

function Newsletter({ variant }) {
  const { t } = useTranslation();
  const location = useLocation();

  // Determine variant based on current route if not provided
  const getVariantFromPath = () => {
    const path = location.pathname;
    if (path.includes('/analysis')) return 'analysis';
    if (path.includes('/stories')) return 'stories';
    if (path.includes('/notable-work')) return 'notable-work';
    if (path.includes('/archive')) return 'archive';
    return 'default';
  };

  const currentVariant = variant || getVariantFromPath();

  const themeClasses = {
    analysis: {
      container: '!bg-blue-100 dark:!bg-blue-900/30',
      button: '!bg-blue-600 hover:!bg-blue-700 dark:!bg-blue-700 dark:hover:!bg-blue-800',
      ring: '!ring-blue-500 dark:!ring-blue-400',
      text: '!text-blue-900 dark:!text-blue-100'
    },
    stories: {
      container: '!bg-amber-100 dark:!bg-amber-900/30',
      button: '!bg-amber-600 hover:!bg-amber-700 dark:!bg-amber-700 dark:hover:!bg-amber-800',
      ring: '!ring-amber-500 dark:!ring-amber-400',
      text: '!text-amber-900 dark:!text-amber-100'
    },
    'notable-work': {
      container: '!bg-purple-100 dark:!bg-purple-900/30',
      button: '!bg-purple-600 hover:!bg-purple-700 dark:!bg-purple-700 dark:hover:!bg-purple-800',
      ring: '!ring-purple-500 dark:!ring-purple-400',
      text: '!text-purple-900 dark:!text-purple-100'
    },
    archive: {
      container: '!bg-emerald-100 dark:!bg-emerald-900/30',
      button: '!bg-emerald-600 hover:!bg-emerald-700 dark:!bg-emerald-700 dark:hover:!bg-emerald-800',
      ring: '!ring-emerald-500 dark:!ring-emerald-400',
      text: '!text-emerald-900 dark:!text-emerald-100'
    },
    default: {
      container: '!bg-slate-100 dark:!bg-slate-900/30',
      button: '!bg-slate-600 hover:!bg-slate-700 dark:!bg-slate-700 dark:hover:!bg-slate-800',
      ring: '!ring-slate-500 dark:!ring-slate-400',
      text: '!text-slate-900 dark:!text-slate-100'
    }
  };

  const theme = themeClasses[currentVariant] || themeClasses.default;

  return (
    <div className={`relative ptc-newsletter ${theme.container} px-6 py-8 md:px-10 rounded-xl shadow-lg dark:shadow-none`}>
      <div className="max-w-2xl mx-auto text-center">
        <h3 className={`text-2xl font-serif font-bold ${theme.text} mb-4`}>
          {t('Stay Updated')}
        </h3>
        <p className={`${theme.text} mb-6 opacity-90`}>
          {t('Subscribe to our newsletter to receive the latest updates and exclusive content')}
        </p>
        <form className="flex flex-col sm:flex-row gap-4 justify-center">
          <input
            type="email"
            placeholder={t('Enter your email')}
            className={`ptc-newsletter-input px-4 py-2 rounded-lg bg-white/80 dark:!bg-gray-800/80 border border-gray-200 dark:!border-gray-700 focus:outline-none focus:ring-2 ${theme.ring} flex-grow max-w-md placeholder-gray-500 dark:!placeholder-gray-400 text-gray-900 dark:!text-gray-100`}
          />
          <button
            type="submit"
            className={`ptc-newsletter-button px-6 py-2 ${theme.button} text-white rounded-lg transition-colors duration-300 shadow-md hover:shadow-lg dark:shadow-none`}
          >
            {t('Subscribe')}
          </button>
        </form>
      </div>
    </div>
  );
}

export default Newsletter; 