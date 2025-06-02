import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { getArticlesByCategory } from '../data/articles';
import ArticleCard from '../components/ArticleCard';
import Newsletter from '../components/Newsletter';
import Pagination from '../components/Pagination';

function AllSportsHub() {
  const [activeTab, setActiveTab] = useState('latest');
  const { t } = useTranslation();
  const articles = getArticlesByCategory('all-sports-hub');
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
          totalPages={Math.ceil(articles.length / articlesPerPage)}
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