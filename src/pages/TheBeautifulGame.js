import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { getArticlesByCategory } from '../data/articles';
import ArticleCard from '../components/ArticleCard';
import Newsletter from '../components/Newsletter';
import Pagination from '../components/Pagination';

function TheBeautifulGame() {
  const [activeTab, setActiveTab] = useState('latest');
  const { t } = useTranslation();
  const articles = getArticlesByCategory('the-beautiful-game');
  const [currentPage, setCurrentPage] = useState(1);
  const articlesPerPage = 6;

  // Get current articles
  const indexOfLastArticle = currentPage * articlesPerPage;
  const indexOfFirstArticle = indexOfLastArticle - articlesPerPage;
  const currentArticles = articles.slice(indexOfFirstArticle, indexOfLastArticle);

  // Change page
  const paginate = (pageNumber) => setCurrentPage(pageNumber);

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-serif font-bold text-green-900 dark:text-green-400 mb-8">
        {t('The Beautiful Game')}
      </h1>

      {/* Tab Navigation */}
      <div className="flex space-x-4 mb-8">
        <button
          onClick={() => setActiveTab('latest')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'latest'
              ? 'bg-green-900 text-white dark:bg-green-800'
              : 'text-gray-600 hover:bg-green-50 dark:text-gray-300 dark:hover:bg-green-900/10'
          }`}
        >
          {t('Latest')}
        </button>
        <button
          onClick={() => setActiveTab('top')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            activeTab === 'top'
              ? 'bg-green-900 text-white dark:bg-green-800'
              : 'text-gray-600 hover:bg-green-50 dark:text-gray-300 dark:hover:bg-green-900/10'
          }`}
        >
          {t('Top')}
        </button>
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
          totalPages={Math.ceil(articles.length / articlesPerPage)}
          onPageChange={paginate}
          variant="the-beautiful-game"
        />
      </div>

      {/* Newsletter */}
      <div className="mt-12">
        <Newsletter variant="the-beautiful-game" />
      </div>
    </div>
  );
}

export default TheBeautifulGame; 