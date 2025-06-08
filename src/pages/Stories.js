import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { FiHeart, FiMessageCircle, FiShare2, FiArrowRight } from 'react-icons/fi';
import { useTranslation } from 'react-i18next';
import { useArticles, getLocalizedArticleContent, categoryTranslations } from '../hooks/useArticles';
import Newsletter from '../components/Newsletter';
import Pagination from '../components/Pagination';

function Stories() {
  const [activeTab, setActiveTab] = useState('latest');
  const [currentPage, setCurrentPage] = useState(1);
  const { t, i18n } = useTranslation();
  const { fetchArticlesByCategory, loading, error, articles: hookArticles } = useArticles();
  const [articles, setArticles] = useState([]);

  useEffect(() => {
    const loadArticles = async () => {
      try {
        // Map 'stories' to 'the-beautiful-game' category
        const categoryArticles = await fetchArticlesByCategory('the-beautiful-game');
        setArticles(categoryArticles || []);
      } catch (error) {
        console.error('Error loading articles:', error);
      }
    };

    loadArticles();
  }, [fetchArticlesByCategory]);

  // Update local articles when hook articles change (global cache updates)
  useEffect(() => {
    if (hookArticles.length > 0) {
      // Filter for 'the-beautiful-game' category only
      const categoryArticles = hookArticles.filter(article => article.category === 'the-beautiful-game');
      setArticles(categoryArticles);
    }
  }, [hookArticles]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-green-600 mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">{t('Loading articles...')}</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-600 dark:text-red-400">{t('Error loading articles')}: {error}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="mt-4 px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
          >
            {t('Retry')}
          </button>
        </div>
      </div>
    );
  }

  // Sorting logic
  const sortedArticles = [...articles].sort((a, b) => {
    if (activeTab === 'top') {
      return (typeof b.likes === 'object' ? (b.likes.count || 0) : (b.likes || 0)) - (typeof a.likes === 'object' ? (a.likes.count || 0) : (a.likes || 0));
    } else {
      // latest
      return new Date(b.date) - new Date(a.date);
    }
  });

  // Pagination logic
  const articlesPerPage = 5;
  const totalPages = Math.ceil(sortedArticles.length / articlesPerPage);
  const paginatedArticles = sortedArticles.slice((currentPage - 1) * articlesPerPage, currentPage * articlesPerPage);

  // Scroll to top on page change
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [currentPage]);

  // Scroll to top on tab change
  useEffect(() => {
    setCurrentPage(1);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [activeTab]);

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-serif font-bold text-gray-900 dark:text-white mb-8">
        {t('Stories')}
      </h1>

      {/* Tab Navigation */}
      <div className="flex space-x-4 mb-8">
        <button
          onClick={() => setActiveTab('latest')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'latest'
              ? 'bg-amber-600 text-white dark:bg-amber-700'
              : 'text-gray-600 hover:bg-amber-50 dark:text-gray-300 dark:hover:bg-gray-800'
          }`}
        >
          {t('Latest')}
        </button>
        <button
          onClick={() => setActiveTab('top')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'top'
              ? 'bg-amber-600 text-white dark:bg-amber-700'
              : 'text-gray-600 hover:bg-amber-50 dark:text-gray-300 dark:hover:bg-gray-800'
          }`}
        >
          {t('Top')}
        </button>
      </div>

      {/* Articles List */}
      <div className="divide-y divide-gray-200 dark:divide-gray-700">
        {paginatedArticles.map((article) => {
          const localizedContent = getLocalizedArticleContent(article, i18n.language);
          const authorName = i18n.language === 'ar' ? 'صدقي بن حوالة' : article.author;
          return (
            <article key={article.id} className="py-8">
              <div className="flex items-start space-x-8">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-4 text-sm mb-2">
                    <span className="text-amber-700 dark:text-amber-400 font-medium">
                      {categoryTranslations[article.category]?.[i18n.language] || article.category}
                    </span>
                    <span className="text-gray-600 dark:text-gray-400">
                      {article.date}
                    </span>
                  </div>
                  
                  <Link to={`/article/${article.slug || article.id}`}>
                    <h2 className="text-2xl font-serif font-bold text-gray-900 dark:text-white mb-3 hover:text-amber-700 dark:hover:text-amber-400 transition-colors">
                      {localizedContent.title}
                    </h2>
                  </Link>
                  
                  <p className="text-gray-600 dark:text-gray-300 mb-4 line-clamp-3 font-sans text-base">
                    {localizedContent.excerpt || localizedContent.content?.[0]?.content}
                  </p>
                  
                  <div className="flex items-center gap-4 text-sm">
                    <span className="text-gray-600 dark:text-gray-400">
                      {authorName}
                    </span>
                    <div className="flex items-center space-x-4 text-gray-500 dark:text-gray-400">
                      <span className="flex items-center">
                        <FiHeart className={i18n.language === 'ar' ? 'mx-2 h-4 w-4' : 'mr-1 h-4 w-4'} />{typeof article.likes === 'object' ? article.likes.count : article.likes}
                      </span>
                      <span className="flex items-center">
                        <FiMessageCircle className={i18n.language === 'ar' ? 'mx-2 h-4 w-4' : 'mr-1 h-4 w-4'} />{article.comments}
                      </span>
                    </div>
                  </div>
                </div>
                
                <div className="flex-shrink-0 w-48 h-32 overflow-hidden rounded-lg">
                  <img
                    src={article.image}
                    alt={localizedContent.title}
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
            </article>
          );
        })}
      </div>

      {/* Pagination Controls */}
      {totalPages > 1 && (
        <Pagination
          currentPage={currentPage}
          totalPages={totalPages}
          onPageChange={setCurrentPage}
          variant="stories"
        />
      )}

      {/* Newsletter Section */}
      <div className="mt-12">
        <Newsletter variant="stories" />
      </div>
    </div>
  );
}

export default Stories; 