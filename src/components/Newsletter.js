import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useLocation } from 'react-router-dom';
import api from '../services/api';

function Newsletter({ variant }) {
  const { t } = useTranslation();
  const location = useLocation();
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState('');

  // Determine variant based on current route if not provided
  const getVariantFromPath = () => {
    const path = location.pathname;
    if (path.includes('/etoile-du-sahel')) return 'etoile-du-sahel';
    if (path.includes('/the-beautiful-game')) return 'the-beautiful-game';
    if (path.includes('/all-sports-hub')) return 'all-sports-hub';
    if (path.includes('/archive')) return 'archive';
    return 'default';
  };

  const currentVariant = variant || getVariantFromPath();

  const themeClasses = {
    'etoile-du-sahel': {
      container: '!bg-red-100 dark:!bg-red-900/30',
      button: '!bg-red-600 hover:!bg-red-700 dark:!bg-red-700 dark:hover:!bg-red-800',
      ring: '!ring-red-500 dark:!ring-red-400',
      text: '!text-red-900 dark:!text-red-100'
    },
    'the-beautiful-game': {
      container: '!bg-green-100 dark:!bg-green-900/30',
      button: '!bg-green-600 hover:!bg-green-700 dark:!bg-green-700 dark:hover:!bg-green-800',
      ring: '!ring-green-500 dark:!ring-green-400',
      text: '!text-green-900 dark:!text-green-100'
    },
    'all-sports-hub': {
      container: '!bg-purple-100 dark:!bg-purple-900/30',
      button: '!bg-purple-600 hover:!bg-purple-700 dark:!bg-purple-700 dark:hover:!bg-purple-800',
      ring: '!ring-purple-500 dark:!ring-purple-400',
      text: '!text-purple-900 dark:!text-purple-100'
    },
    'archive': {
      container: '!bg-yellow-100 dark:!bg-yellow-900/30',
      button: '!bg-yellow-600 hover:!bg-yellow-700 dark:!bg-yellow-700 dark:hover:!bg-yellow-800',
      ring: '!ring-yellow-500 dark:!ring-yellow-400',
      text: '!text-yellow-900 dark:!text-yellow-100'
    },
    default: {
      container: '!bg-gray-100 dark:!bg-gray-900/30',
      button: '!bg-gray-600 hover:!bg-gray-700 dark:!bg-gray-700 dark:hover:!bg-gray-800',
      ring: '!ring-gray-500 dark:!ring-gray-400',
      text: '!text-gray-900 dark:!text-gray-100'
    }
  };

  const theme = themeClasses[currentVariant] || themeClasses.default;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess(false);

    try {
      await api.post('/newsletter/subscribe', {
        email,
        preferences: {
          type: currentVariant !== 'default' ? currentVariant : 'all'
        }
      });

      setSuccess(true);
      setEmail('');
    } catch (err) {
      setError(err.response?.data?.msg || t('Failed to subscribe. Please try again.'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`relative ptc-newsletter ${theme.container} px-6 py-8 md:px-10 rounded-xl shadow-lg dark:shadow-none`}>
      <div className="max-w-2xl mx-auto text-center">
        <h3 className={`text-2xl font-serif font-bold ${theme.text} mb-4`}>
          {t('Stay Updated')}
        </h3>
        <p className={`${theme.text} mb-6 opacity-90`}>
          {t('Subscribe to our newsletter to receive the latest updates and exclusive content')}
        </p>
        
        {success ? (
          <div className={`${theme.text} text-center`}>
            <svg className="w-12 h-12 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-lg font-medium">
              {t('Thank you for subscribing!')}
            </p>
            <p className="mt-2 opacity-90">
              {t('Please check your email to confirm your subscription.')}
            </p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-4 justify-center">
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder={t('Enter your email')}
              className={`ptc-newsletter-input px-4 py-2 rounded-lg bg-white/80 dark:!bg-gray-800/80 border border-gray-200 dark:!border-gray-700 focus:outline-none focus:ring-2 ${theme.ring} flex-grow max-w-md placeholder-gray-500 dark:!placeholder-gray-400 text-gray-900 dark:!text-gray-100`}
              required
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading}
              className={`ptc-newsletter-button px-6 py-2 ${theme.button} text-white rounded-lg transition-colors duration-300 shadow-md hover:shadow-lg dark:shadow-none disabled:opacity-50`}
            >
              {loading ? t('Subscribing...') : t('Subscribe')}
            </button>
          </form>
        )}

        {error && (
          <p className="mt-4 text-red-500 dark:text-red-400">
            {error}
          </p>
        )}
      </div>
    </div>
  );
}

export default Newsletter; 