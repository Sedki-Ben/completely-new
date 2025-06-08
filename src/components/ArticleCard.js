import React from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

const ArticleCard = ({ article, variant = 'default' }) => {
  const { i18n } = useTranslation();
  
  // Get current language content or fallback to English
  const getCurrentLanguageContent = () => {
    const currentLang = i18n.language;
    if (article.translations && article.translations[currentLang]) {
      return article.translations[currentLang];
    }
    // Fallback to English if current language not available
    return article.translations?.en || {
      title: 'Untitled',
      excerpt: 'No excerpt available'
    };
  };

  const localizedContent = getCurrentLanguageContent();

  // Get author name based on language
  const getAuthorName = () => {
    if (i18n.language === 'ar') {
      return 'صدقي بن حوالة';
    }
    return article.author?.name || article.author || 'Anonymous';
  };

  // Check if current language is RTL
  const isRTL = i18n.language === 'ar';

  const themeClasses = {
    'etoile-du-sahel': {
      gradient: 'from-red-500 to-red-700',
      hover: 'hover:bg-red-50 dark:hover:bg-red-900/10',
      border: 'border-red-200 dark:border-red-800',
      text: 'text-red-600 dark:text-red-400',
      icon: 'text-red-500'
    },
    'the-beautiful-game': {
      gradient: 'from-green-500 to-green-700',
      hover: 'hover:bg-green-50 dark:hover:bg-green-900/10',
      border: 'border-green-200 dark:border-green-800',
      text: 'text-green-600 dark:text-green-400',
      icon: 'text-green-500'
    },
    'all-sports-hub': {
      gradient: 'from-purple-500 to-purple-700',
      hover: 'hover:bg-purple-50 dark:hover:bg-purple-900/10',
      border: 'border-purple-200 dark:border-purple-800',
      text: 'text-purple-600 dark:text-purple-400',
      icon: 'text-purple-500'
    },
    'archive': {
      gradient: 'from-yellow-400 to-yellow-600',
      hover: 'hover:bg-yellow-50 dark:hover:bg-yellow-900/10',
      border: 'border-yellow-200 dark:border-yellow-800',
      text: 'text-yellow-600 dark:text-yellow-400',
      icon: 'text-yellow-500'
    },
    default: {
      gradient: 'from-gray-500 to-gray-700',
      hover: 'hover:bg-gray-50 dark:hover:bg-gray-900/10',
      border: 'border-gray-200 dark:border-gray-800',
      text: 'text-gray-600 dark:text-gray-400',
      icon: 'text-gray-500'
    }
  };

  const theme = themeClasses[article.category] || themeClasses.default;

  return (
    <Link 
      to={`/article/${article.slug || article.id}`}
      className={`block bg-white dark:bg-gray-800 rounded-xl shadow-lg hover:shadow-xl dark:shadow-none transition-all duration-300 overflow-hidden border ${theme.border} ${theme.hover}`}
      dir={isRTL ? 'rtl' : 'ltr'}
    >
      {/* Image */}
      <div className="relative h-48 overflow-hidden">
        <img
          src={article.image}
          alt={localizedContent?.title}
          className="w-full h-full object-cover"
        />
      </div>

      {/* Content */}
      <div className="p-6">
        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
          {localizedContent?.title}
        </h3>
        
        <p className="text-gray-600 dark:text-gray-300 mb-4 line-clamp-2">
          {localizedContent?.excerpt}
        </p>

        {/* Meta information */}
        <div className={`flex items-center justify-between ${isRTL ? 'flex-row-reverse' : ''}`}>
          <div className={`flex items-center ${isRTL ? 'space-x-reverse space-x-2' : 'space-x-2'}`}>
            <img
              src={article.authorImage}
              alt={getAuthorName()}
              className="w-8 h-8 rounded-full object-cover"
              onError={(e) => {
                e.target.src = '/src/assets/images/bild3.jpg'; // Fallback image
              }}
            />
            <span className="text-sm text-gray-600 dark:text-gray-400">
              {getAuthorName()}
            </span>
          </div>
          
          <div className={`flex items-center ${isRTL ? 'space-x-reverse space-x-4' : 'space-x-4'}`}>
            <span className={`text-sm ${theme.text} flex items-center ${isRTL ? 'space-x-reverse space-x-1' : 'space-x-1'}`}>
              <span>{typeof article.likes === 'object' ? (article.likes.count || 0) : (article.likes || 0)}</span>
              <span>❤️</span>
            </span>
            <span className={`text-sm ${theme.text} flex items-center ${isRTL ? 'space-x-reverse space-x-1' : 'space-x-1'}`}>
              <span>{article.comments || 0}</span>
              <span>💬</span>
            </span>
          </div>
        </div>
      </div>
    </Link>
  );
};

export default ArticleCard; 