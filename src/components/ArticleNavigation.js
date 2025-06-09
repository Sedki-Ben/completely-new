import React from 'react';
import { Link } from 'react-router-dom';
import { FiChevronLeft, FiChevronRight, FiClock, FiStar } from 'react-icons/fi';
import { GiBoxingGlove } from 'react-icons/gi';
import { useTranslation } from 'react-i18next';
import { getLocalizedArticleContent } from '../hooks/useArticles';

const ArticleNavigation = ({ currentArticle, previousArticle, nextArticle, loading = false }) => {
  const { t, i18n } = useTranslation();
  const isRTL = i18n.language === 'ar';

  // Debug logging
  console.log('ArticleNavigation render:', { 
    loading, 
    hasPrevious: !!previousArticle, 
    hasNext: !!nextArticle,
    currentArticleId: currentArticle?.id || currentArticle?._id 
  });

  // Get theme colors based on category
  const getThemeColors = (category) => {
    switch (category) {
      case 'etoile-du-sahel':
        return {
          primary: 'from-red-500 to-red-600',
          light: 'from-red-50 to-red-100',
          text: 'text-red-700',
          border: 'border-red-200',
          icon: 'text-red-500',
          darkGradient: 'dark:from-red-900/20 dark:to-red-800/20',
          darkBorder: 'dark:border-red-800/30'
        };
      case 'the-beautiful-game':
        return {
          primary: 'from-green-500 to-emerald-600',
          light: 'from-green-50 to-emerald-100',
          text: 'text-green-700',
          border: 'border-green-200',
          icon: 'text-green-500',
          darkGradient: 'dark:from-green-900/20 dark:to-emerald-800/20',
          darkBorder: 'dark:border-green-800/30'
        };
      case 'all-sports-hub':
        return {
          primary: 'from-purple-500 to-indigo-600',
          light: 'from-purple-50 to-indigo-100',
          text: 'text-purple-700',
          border: 'border-purple-200',
          icon: 'text-purple-500',
          darkGradient: 'dark:from-purple-900/20 dark:to-indigo-800/20',
          darkBorder: 'dark:border-purple-800/30'
        };
      default:
        return {
          primary: 'from-gray-500 to-gray-600',
          light: 'from-gray-50 to-gray-100',
          text: 'text-gray-700',
          border: 'border-gray-200',
          icon: 'text-gray-500',
          darkGradient: 'dark:from-gray-900/20 dark:to-gray-800/20',
          darkBorder: 'dark:border-gray-800/30'
        };
    }
  };

  // Get category icon
  const getCategoryIcon = (category) => {
    switch (category) {
      case 'etoile-du-sahel':
        return <FiStar className="w-5 h-5" />;
      case 'the-beautiful-game':
        return <span className="text-lg">⚽</span>;
      case 'all-sports-hub':
        return <GiBoxingGlove className="w-5 h-5" />;
      default:
        return <FiClock className="w-5 h-5" />;
    }
  };

  // Get category translation
  const getCategoryName = (category) => {
    const translations = {
      'etoile-du-sahel': {
        en: 'Etoile Du Sahel',
        fr: 'Étoile Du Sahel',
        ar: 'النجم الساحلي'
      },
      'the-beautiful-game': {
        en: 'The Beautiful Game',
        fr: 'Le Beau Jeu',
        ar: 'اللعبة الجميلة'
      },
      'all-sports-hub': {
        en: 'All-Sports Hub',
        fr: 'Centre Omnisports',
        ar: 'مركز كل الرياضات'
      }
    };
    return translations[category]?.[i18n.language] || category;
  };

  const theme = getThemeColors(currentArticle.category);

  // Show loading state only if we're actually loading AND have no navigation articles yet
  if (loading && !previousArticle && !nextArticle) {
    return (
      <div className={`max-w-4xl mx-auto mt-16 mb-8 ${isRTL ? 'text-right' : 'text-left'}`}>
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className={`w-12 h-12 bg-gradient-to-br ${theme.primary} rounded-xl flex items-center justify-center shadow-lg animate-pulse`}>
              {getCategoryIcon(currentArticle.category)}
            </div>
            <h3 className={`text-2xl font-serif font-bold ${theme.text} dark:text-gray-200`}>
              {t('More from')} {getCategoryName(currentArticle.category)}
            </h3>
          </div>
          <div className={`w-20 h-1 bg-gradient-to-r ${theme.primary} mx-auto rounded-full animate-pulse`}></div>
        </div>
        <div className="grid gap-6 grid-cols-1 md:grid-cols-2">
          {[1, 2].map((index) => (
            <div key={index} className={`bg-gradient-to-br ${theme.light} ${theme.darkGradient} backdrop-blur-lg rounded-2xl p-6 shadow-lg border ${theme.border} ${theme.darkBorder} animate-pulse`}>
              <div className="flex items-center gap-3 mb-4">
                <div className={`w-6 h-6 bg-gray-300 dark:bg-gray-600 rounded`}></div>
                <div className={`h-4 bg-gray-300 dark:bg-gray-600 rounded w-24`}></div>
              </div>
              <div className="flex gap-4">
                <div className="w-20 h-20 bg-gray-300 dark:bg-gray-600 rounded-lg"></div>
                <div className="flex-1">
                  <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded mb-2"></div>
                  <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded w-3/4 mb-2"></div>
                  <div className="h-3 bg-gray-300 dark:bg-gray-600 rounded w-1/2"></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Don't render if no navigation is possible and not loading
  if (!previousArticle && !nextArticle) {
    return null;
  }

  return (
    <div className={`max-w-4xl mx-auto mt-16 mb-8 ${isRTL ? 'text-right' : 'text-left'}`} dir={isRTL ? 'rtl' : 'ltr'}>
      {/* Section Header */}
      <div className="text-center mb-8">
        <div className="flex items-center justify-center gap-3 mb-4">
          <div className={`w-12 h-12 bg-gradient-to-br ${theme.primary} rounded-xl flex items-center justify-center shadow-lg`}>
            {getCategoryIcon(currentArticle.category)}
          </div>
          <h3 className={`text-2xl font-serif font-bold ${theme.text} dark:text-gray-200`}>
            {t('More from')} {getCategoryName(currentArticle.category)}
          </h3>
        </div>
        <div className={`w-20 h-1 bg-gradient-to-r ${theme.primary} mx-auto rounded-full`}></div>
      </div>

      {/* Navigation Cards */}
      <div className="grid gap-6 grid-cols-1 md:grid-cols-2">
        
        {/* Previous Article - Always takes the left/first position */}
        {previousArticle ? (
          <Link 
            to={`/article/${previousArticle.slug || previousArticle.id}`}
            className={`group relative bg-gradient-to-br ${theme.light} ${theme.darkGradient} backdrop-blur-lg rounded-2xl p-6 shadow-lg border ${theme.border} ${theme.darkBorder} hover:shadow-2xl hover:scale-[1.02] transition-all duration-500 overflow-hidden`}
          >
            {/* Decorative Elements */}
            <div className={`absolute top-0 right-0 w-20 h-20 bg-gradient-to-br ${theme.primary} opacity-10 rounded-full transform translate-x-10 -translate-y-10`}></div>
            <div className={`absolute bottom-0 left-0 w-16 h-16 bg-gradient-to-tr ${theme.primary} opacity-10 rounded-full transform -translate-x-8 translate-y-8`}></div>
            
            <div className="relative">
              {/* Direction Indicator */}
              <div className={`flex items-center gap-3 mb-4 ${isRTL ? 'flex-row-reverse' : ''}`}>
                {isRTL ? (
                  <FiChevronRight className={`w-6 h-6 ${theme.icon} group-hover:translate-x-1 transition-transform duration-300`} />
                ) : (
                  <FiChevronLeft className={`w-6 h-6 ${theme.icon} group-hover:-translate-x-1 transition-transform duration-300`} />
                )}
                <span className={`text-sm font-medium ${theme.text} dark:text-gray-400 uppercase tracking-wide`}>
                  {t('Previous Article')}
                </span>
              </div>
              
              {/* Article Preview */}
              <div className={`flex gap-4 ${isRTL ? 'flex-row-reverse' : ''}`}>
                <div className="flex-shrink-0 w-20 h-20 rounded-lg overflow-hidden shadow-md">
                  <img
                    src={previousArticle.image}
                    alt={getLocalizedArticleContent(previousArticle, i18n.language)?.title}
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <h4 className={`font-serif font-bold text-gray-900 dark:text-white mb-2 line-clamp-2 group-hover:${theme.text.replace('text-', 'text-')} transition-colors duration-300`}>
                    {getLocalizedArticleContent(previousArticle, i18n.language)?.title}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {previousArticle.date}
                  </p>
                </div>
              </div>
            </div>
          </Link>
        ) : (
          /* Empty space for previous article when it doesn't exist */
          <div></div>
        )}

        {/* Next Article - Always takes the right/second position */}
        {nextArticle ? (
          <Link 
            to={`/article/${nextArticle.slug || nextArticle.id}`}
            className={`group relative bg-gradient-to-br ${theme.light} ${theme.darkGradient} backdrop-blur-lg rounded-2xl p-6 shadow-lg border ${theme.border} ${theme.darkBorder} hover:shadow-2xl hover:scale-[1.02] transition-all duration-500 overflow-hidden`}
          >
            {/* Decorative Elements */}
            <div className={`absolute top-0 left-0 w-20 h-20 bg-gradient-to-br ${theme.primary} opacity-10 rounded-full transform -translate-x-10 -translate-y-10`}></div>
            <div className={`absolute bottom-0 right-0 w-16 h-16 bg-gradient-to-tl ${theme.primary} opacity-10 rounded-full transform translate-x-8 translate-y-8`}></div>
            
            <div className="relative">
              {/* Direction Indicator */}
              <div className={`flex items-center gap-3 mb-4 ${isRTL ? 'flex-row-reverse justify-end' : 'justify-end'}`}>
                <span className={`text-sm font-medium ${theme.text} dark:text-gray-400 uppercase tracking-wide`}>
                  {t('Next Article')}
                </span>
                {isRTL ? (
                  <FiChevronLeft className={`w-6 h-6 ${theme.icon} group-hover:-translate-x-1 transition-transform duration-300`} />
                ) : (
                  <FiChevronRight className={`w-6 h-6 ${theme.icon} group-hover:translate-x-1 transition-transform duration-300`} />
                )}
              </div>
              
              {/* Article Preview */}
              <div className={`flex gap-4 ${isRTL ? 'flex-row-reverse' : ''}`}>
                <div className="flex-1 min-w-0">
                  <h4 className={`font-serif font-bold text-gray-900 dark:text-white mb-2 line-clamp-2 group-hover:${theme.text.replace('text-', 'text-')} transition-colors duration-300 ${isRTL ? 'text-right' : 'text-right'}`}>
                    {getLocalizedArticleContent(nextArticle, i18n.language)?.title}
                  </h4>
                  <p className={`text-sm text-gray-600 dark:text-gray-400 ${isRTL ? 'text-right' : 'text-right'}`}>
                    {nextArticle.date}
                  </p>
                </div>
                <div className="flex-shrink-0 w-20 h-20 rounded-lg overflow-hidden shadow-md">
                  <img
                    src={nextArticle.image}
                    alt={getLocalizedArticleContent(nextArticle, i18n.language)?.title}
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
            </div>
          </Link>
        ) : (
          /* Empty space for next article when it doesn't exist */
          <div></div>
        )}
      </div>
    </div>
  );
};

export default ArticleNavigation; 