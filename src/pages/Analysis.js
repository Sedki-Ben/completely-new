import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { useArticles } from '../hooks/useArticles';
import ArticleCard from '../components/ArticleCard';
import Newsletter from '../components/Newsletter';
import Pagination from '../components/Pagination';

function Analysis() {
  const { t, i18n } = useTranslation();
  const { fetchArticlesByCategory, loading, error, articles: hookArticles } = useArticles();
  const [articles, setArticles] = useState([]);
  const [activeTab, setActiveTab] = useState('latest');
  const [currentPage, setCurrentPage] = useState(1);
  const articlesPerPage = 6;

  useEffect(() => {
    const loadArticles = async () => {
      try {
        const categoryArticles = await fetchArticlesByCategory('etoile-du-sahel');
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
      // Filter for 'etoile-du-sahel' category only
      const categoryArticles = hookArticles.filter(article => article.category === 'etoile-du-sahel');
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
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-red-600 mx-auto"></div>
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
            className="mt-4 px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
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
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-serif font-bold text-gray-900 dark:text-white mb-8">
        {t('Etoile Du Sahel')}
      </h1>

      {/* Tab Navigation */}
      <div className="flex space-x-4 mb-8">
        <button
          onClick={() => setActiveTab('latest')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'latest'
              ? 'bg-blue-900 text-white dark:bg-blue-800'
              : 'text-gray-600 hover:bg-blue-50 dark:text-gray-300 dark:hover:bg-gray-800'
          }`}
        >
          {t('Latest')}
        </button>
        <button
          onClick={() => setActiveTab('top')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'top'
              ? 'bg-blue-900 text-white dark:bg-blue-800'
              : 'text-gray-600 hover:bg-blue-50 dark:text-gray-300 dark:hover:bg-gray-800'
          }`}
        >
          {t('Top')}
        </button>
      </div>

      {/* Articles Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-8">
        {currentArticles.map((article) => (
          <ArticleCard key={article.id} article={article} variant="etoile-du-sahel" />
        ))}
      </div>

      {/* Pagination */}
      <div className="mt-8">
        <Pagination
          currentPage={currentPage}
          totalPages={Math.ceil(sortedArticles.length / articlesPerPage)}
          onPageChange={paginate}
          variant="etoile-du-sahel"
        />
      </div>

      {/* Newsletter */}
      <div className="mt-12">
        <Newsletter variant="etoile-du-sahel" />
      </div>
    </div>
  );
}

export default Analysis; 