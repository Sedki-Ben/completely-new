import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { useArticles } from '../hooks/useArticles';
import ArticleCard from '../components/ArticleCard';
import Newsletter from '../components/Newsletter';
import Pagination from '../components/Pagination';

function TheBeautifulGame() {
  const { t } = useTranslation();
  const { fetchArticlesByCategory, loading, error, articles: hookArticles } = useArticles();
  const [articles, setArticles] = useState([]);
  const [activeTab, setActiveTab] = useState('latest');
  const [currentPage, setCurrentPage] = useState(1);
  const articlesPerPage = 6;

  useEffect(() => {
    const loadArticles = async () => {
      try {
        // Try cache first, then fetch if not available
        const categoryArticles = await fetchArticlesByCategory('the-beautiful-game', true);
        setArticles(categoryArticles || []);
        
        // If we got very few articles from cache, refresh from API
        if (!categoryArticles || categoryArticles.length < 3) {
          console.log('Cache empty or insufficient, refreshing from API');
          const freshArticles = await fetchArticlesByCategory('the-beautiful-game', false);
          setArticles(freshArticles || []);
        }
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

  // Reset page when tab changes
  useEffect(() => {
    setCurrentPage(1);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [activeTab]);

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
    if (activeTab === 'most-liked') {
      return (typeof b.likes === 'object' ? (b.likes.count || 0) : (b.likes || 0)) - (typeof a.likes === 'object' ? (a.likes.count || 0) : (a.likes || 0));
    } else if (activeTab === 'most-discussed') {
      return (typeof b.comments === 'object' ? (b.comments.count || 0) : (b.comments || 0)) - (typeof a.comments === 'object' ? (a.comments.count || 0) : (a.comments || 0));
    } else {
      // latest - sort by raw date for proper chronological order
      return new Date(b.rawDate || b.date) - new Date(a.rawDate || a.date);
    }
  });

  // Get current articles
  const indexOfLastArticle = currentPage * articlesPerPage;
  const indexOfFirstArticle = indexOfLastArticle - articlesPerPage;
  const currentArticles = sortedArticles.slice(indexOfFirstArticle, indexOfLastArticle);

  // Change page
  const paginate = (pageNumber) => setCurrentPage(pageNumber);

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-teal-50 to-emerald-100 dark:from-gray-900 dark:via-green-900/20 dark:to-gray-900">
            <div className="container mx-auto px-4 py-8">
        {/* Hero Section */}
        <div className="text-center mb-16 relative">
          <div className="absolute inset-0 bg-green-500/5 dark:bg-green-400/10 rounded-full blur-3xl transform -translate-y-12"></div>
          <h1 className="text-6xl font-serif font-bold bg-gradient-to-r from-green-600 via-green-700 to-green-800 dark:from-green-400 dark:via-green-300 dark:to-green-200 bg-clip-text text-transparent mb-6 relative">
            {t('The Beautiful Game')}
          </h1>
          <div className="w-24 h-1 bg-gradient-to-r from-green-500 to-green-700 mx-auto rounded-full"></div>
        </div>

      {/* Tab Navigation */}
      <div className="max-w-4xl mx-auto mb-12">
        <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-lg rounded-2xl p-6 shadow-lg border border-green-100 dark:border-green-800/30 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-green-500/10 to-transparent rounded-full transform translate-x-10 -translate-y-10"></div>
          <div className="flex gap-4 justify-center relative">
            <button
              onClick={() => setActiveTab('latest')}
              className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                activeTab === 'latest'
                  ? 'bg-gradient-to-r from-green-500 to-green-600 text-white shadow-lg'
                  : 'text-gray-600 hover:bg-green-50 dark:text-gray-300 dark:hover:bg-green-900/10'
              }`}
            >
              {t('Latest')}
            </button>
            <button
              onClick={() => setActiveTab('most-liked')}
              className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                activeTab === 'most-liked'
                  ? 'bg-gradient-to-r from-green-500 to-green-600 text-white shadow-lg'
                  : 'text-gray-600 hover:bg-green-50 dark:text-gray-300 dark:hover:bg-green-900/10'
              }`}
            >
              {t('Most Liked')}
            </button>
            <button
              onClick={() => setActiveTab('most-discussed')}
              className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                activeTab === 'most-discussed'
                  ? 'bg-gradient-to-r from-green-500 to-green-600 text-white shadow-lg'
                  : 'text-gray-600 hover:bg-green-50 dark:text-gray-300 dark:hover:bg-green-900/10'
              }`}
            >
              {t('Most Discussed')}
            </button>
          </div>
        </div>
      </div>

      {/* Articles Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-8">
        {currentArticles.map((article) => (
          <ArticleCard key={article.id} article={article} variant="the-beautiful-game" />
        ))}
      </div>

      {/* Pagination */}
      <div className="mt-8">
        <Pagination
          currentPage={currentPage}
          totalPages={Math.ceil(sortedArticles.length / articlesPerPage)}
          onPageChange={paginate}
          variant="the-beautiful-game"
        />
      </div>

      {/* Newsletter */}
      <div className="mt-12">
        <Newsletter variant="the-beautiful-game" />
      </div>
      </div>
    </div>
  );
}

export default TheBeautifulGame; 