import React from 'react';
import { FiHeart, FiMessageCircle, FiShare2, FiBookmark } from 'react-icons/fi';
import { useTranslation } from 'react-i18next';
import { getLocalizedArticleContent, categoryTranslations } from '../data/articles';
import Newsletter from './Newsletter';

function Article({ article }) {
  const { i18n, t } = useTranslation();
  
  if (!article) {
    return <div>{t('No articles available')}</div>;
  }

  const localizedContent = getLocalizedArticleContent(article, i18n.language);

  // Theme colors based on article category
  const themeColors = {
    'etoile-du-sahel': {
      light: 'bg-red-100 text-red-900 dark:bg-red-900 dark:text-red-100',
      border: 'border-red-900 dark:border-red-600',
      hover: 'hover:text-red-900 dark:hover:text-red-400',
      icon: 'text-red-500'
    },
    'the-beautiful-game': {
      light: 'bg-green-100 text-green-900 dark:bg-green-900 dark:text-green-100',
      border: 'border-green-900 dark:border-green-600',
      hover: 'hover:text-green-900 dark:hover:text-green-400',
      icon: 'text-green-500'
    },
    'all-sports-hub': {
      light: 'bg-purple-100 text-purple-900 dark:bg-purple-900 dark:text-purple-100',
      border: 'border-purple-900 dark:border-purple-600',
      hover: 'hover:text-purple-900 dark:hover:text-purple-400',
      icon: 'text-purple-500'
    },
    'archive': {
      light: 'bg-yellow-100 text-yellow-900 dark:bg-yellow-900 dark:text-yellow-100',
      border: 'border-yellow-900 dark:border-yellow-600',
      hover: 'hover:text-yellow-900 dark:hover:text-yellow-400',
      icon: 'text-yellow-500'
    },
    default: {
      light: 'bg-gray-100 text-gray-900 dark:bg-gray-900 dark:text-gray-100',
      border: 'border-gray-900 dark:border-gray-600',
      hover: 'hover:text-gray-900 dark:hover:text-gray-400',
      icon: 'text-gray-500'
    }
  };

  const theme = themeColors[article.category] || themeColors.default;

  const authorName = i18n.language === 'ar' ? 'صدقي بن حوالة' : article.author;

  return (
    <article className="max-w-4xl mx-auto bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
      {/* Hero Image */}
      <div className="relative h-[40vh] md:h-[50vh]">
        <img
          src={article.image}
          alt={localizedContent.title}
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
      </div>

      {/* Article Content */}
      <div className="px-6 py-8 md:px-10">
        {/* Article Header */}
        <div className="mb-8">
          <div className="flex items-center gap-4 mb-4">
            <span className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium ${theme.light}`}>
              <FiBookmark className={theme.icon} />
              {categoryTranslations[article.category]?.[i18n.language] || article.category}
            </span>
            <span className="text-gray-600 dark:text-gray-400">
              {article.date}
            </span>
          </div>
          <h1 className="text-3xl md:text-4xl lg:text-5xl font-serif font-bold text-gray-900 dark:text-white mb-4">
            {localizedContent.title}
          </h1>
          <div className="flex items-center gap-4">
            <img
              src={article.authorImage || 'https://via.placeholder.com/40'}
              alt={article.author}
              className="w-12 h-12 rounded-full object-cover object-center border-2 border-gray-200 dark:border-gray-700 mr-3"
            />
            <span className="font-medium text-gray-900 dark:text-white ml-2">
              {authorName}
            </span>
          </div>
        </div>

        {/* Article Body */}
        <div className="prose prose-lg dark:prose-invert max-w-none">
          {localizedContent.content.map((section, index) => (
            <div key={index} className="mb-8">
              {section.type === 'paragraph' && (
                <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
                  {section.content}
                </p>
              )}
              {section.type === 'subheading' && (
                <h2 className="text-2xl font-serif font-bold text-gray-900 dark:text-white mt-8 mb-4">
                  {section.content}
                </h2>
              )}
              {section.type === 'image' && (
                <figure className="my-8">
                  <img
                    src={section.url}
                    alt={section.caption}
                    className="w-full rounded-lg shadow-lg"
                  />
                  {section.caption && (
                    <figcaption className="mt-2 text-center text-sm text-gray-600 dark:text-gray-400">
                      {section.caption}
                    </figcaption>
                  )}
                </figure>
              )}
              {section.type === 'quote' && (
                <blockquote className={`border-l-4 ${theme.border} pl-4 italic my-6`}>
                  {section.content}
                  {section.author && (
                    <footer className="text-gray-600 dark:text-gray-400 mt-2">
                      — {section.author}
                    </footer>
                  )}
                </blockquote>
              )}
            </div>
          ))}
        </div>

        {/* Article Footer */}
        <div className="mt-12 pt-6 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-8">
              <button className={`flex items-center gap-3 text-gray-600 dark:text-gray-400 ${theme.hover}`}>
                <FiHeart className={`w-5 h-5 ${theme.icon}`} />
                <span>{article.likes}</span>
              </button>
              <button className={`flex items-center gap-3 text-gray-600 dark:text-gray-400 ${theme.hover}`}>
                <FiMessageCircle className={`w-5 h-5 ${theme.icon}`} />
                <span>{article.comments}</span>
              </button>
            </div>
            <button className={`flex items-center gap-3 text-gray-600 dark:text-gray-400 ${theme.hover}`}>
              <FiShare2 className={`w-5 h-5 ${theme.icon}`} />
              <span>{t('Share')}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Newsletter Section */}
      <div className="mt-12">
        <Newsletter variant={article.category} />
      </div>
    </article>
  );
}

export default Article; 