import React from 'react';
import { FiHeart, FiMessageCircle, FiShare2 } from 'react-icons/fi';
import { useTranslation } from 'react-i18next';
import { getLocalizedArticleContent } from '../data/articles';
import Newsletter from './Newsletter';

function Article({ article }) {
  const { i18n, t } = useTranslation();
  
  if (!article) {
    return <div>{t('No articles available')}</div>;
  }

  const localizedContent = getLocalizedArticleContent(article, i18n.language);

  // Theme colors based on article category
  const themeColors = {
    analysis: {
      light: 'bg-blue-100 text-blue-900 dark:bg-blue-900 dark:text-blue-100',
      border: 'border-blue-900 dark:border-blue-600',
      hover: 'hover:text-blue-900 dark:hover:text-blue-400'
    },
    stories: {
      light: 'bg-amber-100 text-amber-900 dark:bg-amber-900 dark:text-amber-100',
      border: 'border-amber-900 dark:border-amber-600',
      hover: 'hover:text-amber-900 dark:hover:text-amber-400'
    },
    'notable-work': {
      light: 'bg-purple-100 text-purple-900 dark:bg-purple-900 dark:text-purple-100',
      border: 'border-purple-900 dark:border-purple-600',
      hover: 'hover:text-purple-900 dark:hover:text-purple-400'
    },
    archive: {
      light: 'bg-emerald-100 text-emerald-900 dark:bg-emerald-900 dark:text-emerald-100',
      border: 'border-emerald-900 dark:border-emerald-600',
      hover: 'hover:text-emerald-900 dark:hover:text-emerald-400'
    },
    default: {
      light: 'bg-slate-100 text-slate-900 dark:bg-slate-900 dark:text-slate-100',
      border: 'border-slate-900 dark:border-slate-600',
      hover: 'hover:text-slate-900 dark:hover:text-slate-400'
    }
  };

  const theme = themeColors[article.category] || themeColors.default;

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
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${theme.light}`}>
              {t(article.category)}
            </span>
            <span className="text-gray-600 dark:text-gray-400">
              {article.date}
            </span>
          </div>
          <h1 className="text-3xl md:text-4xl lg:text-5xl font-serif font-bold text-gray-900 dark:text-white mb-4">
            {localizedContent.title}
          </h1>
          <div className="flex items-center gap-4">
            <div className="flex items-center">
              <img
                src={article.authorImage || 'https://via.placeholder.com/40'}
                alt={article.author}
                className="w-10 h-10 rounded-full mr-3"
              />
              <span className="font-medium text-gray-900 dark:text-white">
                {t('By')} {article.author}
              </span>
            </div>
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
            <div className="flex items-center space-x-6">
              <button className={`flex items-center space-x-2 text-gray-600 dark:text-gray-400 ${theme.hover}`}>
                <FiHeart className="w-5 h-5" />
                <span>{article.likes}</span>
              </button>
              <button className={`flex items-center space-x-2 text-gray-600 dark:text-gray-400 ${theme.hover}`}>
                <FiMessageCircle className="w-5 h-5" />
                <span>{article.comments}</span>
              </button>
            </div>
            <button className={`flex items-center space-x-2 text-gray-600 dark:text-gray-400 ${theme.hover}`}>
              <FiShare2 className="w-5 h-5" />
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