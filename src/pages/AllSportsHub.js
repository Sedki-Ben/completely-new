import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { useArticles } from '../hooks/useArticles';
import ArticleCard from '../components/ArticleCard';
import Newsletter from '../components/Newsletter';
import Pagination from '../components/Pagination';

function AllSportsHub() {
  const { t } = useTranslation();
  const { fetchArticlesByCategory, loading, error } = useArticles();
  const [articles, setArticles] = useState([]);
  const [activeTab, setActiveTab] = useState('latest');
  const [currentPage, setCurrentPage] = useState(1);
  const articlesPerPage = 6;

  useEffect(() => {
    const loadArticles = async () => {
      try {
        const categoryArticles = await fetchArticlesByCategory('all-sports-hub');
        setArticles(categoryArticles || []);
      } catch (error) {
        console.error('Error loading articles:', error);
      }
    };

    loadArticles();
  }, [fetchArticlesByCategory]);

  // Reset page when tab changes
  useEffect(() => {
    setCurrentPage(1);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [activeTab]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-purple-600 mx-auto"></div>
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
            className="mt-4 px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600"
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
      return b.likes - a.likes;
    } else {
      // latest - sort by date
      return new Date(b.date) - new Date(a.date);
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
      <h1 className="text-4xl font-serif font-bold text-purple-900 dark:text-purple-400 mb-8">
        {t('All-Sports Hub')}
      </h1>

      {/* Tab Navigation */}
      <div className="flex space-x-4 mb-8">
        <button
          onClick={() => setActiveTab('latest')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'latest'
              ? 'bg-purple-900 text-white dark:bg-purple-800'
              : 'text-gray-600 hover:bg-purple-50 dark:text-gray-300 dark:hover:bg-purple-900/10'
          }`}
        >
          {t('Latest')}
        </button>
        <button
          onClick={() => setActiveTab('top')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'top'
              ? 'bg-purple-900 text-white dark:bg-purple-800'
              : 'text-gray-600 hover:bg-purple-50 dark:text-gray-300 dark:hover:bg-purple-900/10'
          }`}
        >
          {t('Top')}
        </button>
      </div>

      {/* Articles Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-8">
        {currentArticles.map((article) => (
          <ArticleCard key={article.id} article={article} variant="all-sports-hub" />
        ))}
      </div>

      {/* Pagination */}
      <div className="mt-8">
        <Pagination
          currentPage={currentPage}
          totalPages={Math.ceil(sortedArticles.length / articlesPerPage)}
          onPageChange={paginate}
          variant="all-sports-hub"
        />
      </div>

      {/* Newsletter */}
      <div className="mt-12">
        <Newsletter variant="all-sports-hub" />
      </div>
    </div>
  );
}

export default AllSportsHub; 