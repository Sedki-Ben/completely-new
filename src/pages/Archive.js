import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { FiHeart, FiMessageCircle } from 'react-icons/fi';
import { useTranslation } from 'react-i18next';
import { useArticles, getLocalizedArticleContent, categoryTranslations } from '../hooks/useArticles';
import Newsletter from '../components/Newsletter';
import Pagination from '../components/Pagination';

function Archive() {
  const { t, i18n } = useTranslation();
  const { fetchAllArticles, loading, error } = useArticles();
  const [allArticles, setAllArticles] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [currentPage, setCurrentPage] = useState(1);

  useEffect(() => {
    const loadArticles = async () => {
      try {
        const articles = await fetchAllArticles();
        setAllArticles(articles || []);
      } catch (error) {
        console.error('Error loading articles:', error);
      }
    };

    loadArticles();
  }, [fetchAllArticles]);

  // Scroll to top on page change
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [currentPage]);

  // Scroll to top and reset page on year change
  useEffect(() => {
    setCurrentPage(1);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [selectedCategory]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-yellow-600 mx-auto"></div>
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
            className="mt-4 px-4 py-2 bg-yellow-500 text-white rounded hover:bg-yellow-600"
          >
            {t('Retry')}
          </button>
        </div>
      </div>
    );
  }

  const articles = allArticles || [];

  // Safely get years from articles with proper date parsing
  const years = [...new Set(articles.map(article => {
    try {
      return new Date(article.date).getFullYear();
    } catch (e) {
      return null;
    }
  }).filter(year => year !== null))].sort((a, b) => b - a);

  // Filter articles by selected year with safe date parsing
  const filteredArticles = selectedCategory === 'all' 
    ? articles 
    : articles.filter(article => {
        try {
          return new Date(article.date).getFullYear() === parseInt(selectedCategory);
        } catch (e) {
          return false;
        }
      });

  // Pagination logic
  const articlesPerPage = 5;
  const totalPages = Math.ceil(filteredArticles.length / articlesPerPage);
  const paginatedArticles = filteredArticles.slice((currentPage - 1) * articlesPerPage, currentPage * articlesPerPage);

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-serif font-bold text-yellow-900 dark:text-yellow-400 mb-8">
        {t('Archive')}
      </h1>

      {/* Year Filter */}
      <div className="flex flex-wrap gap-2 mb-8">
        <button
          onClick={() => setSelectedCategory('all')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            selectedCategory === 'all'
              ? 'bg-yellow-900 text-white dark:bg-yellow-800'
              : 'text-gray-600 hover:bg-yellow-50 dark:text-gray-300 dark:hover:bg-yellow-900/10'
          }`}
        >
          {t('All Years')}
        </button>
        {years.map(year => (
          <button
            key={year}
            onClick={() => setSelectedCategory(year.toString())}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedCategory === year.toString()
                ? 'bg-yellow-900 text-white dark:bg-yellow-800'
                : 'text-gray-600 hover:bg-yellow-50 dark:text-gray-300 dark:hover:bg-yellow-900/10'
            }`}
          >
            {year}
          </button>
        ))}
      </div>

      {/* Articles List */}
      <div className="divide-y divide-gray-200 dark:divide-gray-700">
        {paginatedArticles.map((article) => {
          const localizedContent = getLocalizedArticleContent(article, i18n.language);
          const authorName = i18n.language === 'ar' ? 'صدقي بن حوالة' : article.author;
          // Get the appropriate color based on category
          const categoryColors = {
            'etoile-du-sahel': 'text-red-700 dark:text-red-400',
            'the-beautiful-game': 'text-green-700 dark:text-green-400',
            'all-sports-hub': 'text-purple-700 dark:text-purple-400',
            'archive': 'text-yellow-700 dark:text-yellow-400'
          };
          const categoryColor = categoryColors[article.category] || categoryColors.archive;

          return (
            <article key={article.id} className="py-8">
              <div className="flex items-start space-x-8">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-4 text-sm mb-2">
                    <span className={`font-medium ${categoryColor}`}>
                      {categoryTranslations[article.category]?.[i18n.language] || article.category}
                    </span>
                    <span className="text-gray-600 dark:text-gray-400">
                      {article.date}
                    </span>
                  </div>
                  
                  <Link to={`/article/${article.slug || article.id}`}>
                    <h2 className="text-2xl font-serif font-bold text-gray-900 dark:text-white mb-3 hover:text-yellow-700 dark:hover:text-yellow-400 transition-colors">
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
                        <FiHeart className={i18n.language === 'ar' ? 'mx-2 h-4 w-4' : 'mr-1 h-4 w-4'} />{article.likes}
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
        {filteredArticles.length === 0 && (
          <div className="text-center py-8 text-gray-600 dark:text-gray-400">
            {selectedCategory === 'all' 
              ? t('No articles available yet.')
              : t('No articles available for {{year}}.', { year: selectedCategory })}
          </div>
        )}
      </div>

      {/* Pagination Controls */}
      {totalPages > 1 && (
        <Pagination
          currentPage={currentPage}
          totalPages={totalPages}
          onPageChange={setCurrentPage}
          variant="archive"
        />
      )}

      {/* Newsletter Section */}
      <div className="mt-12">
        <Newsletter variant="archive" />
      </div>
    </div>
  );
}

export default Archive; 