import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { FiHeart, FiMessageCircle, FiCalendar } from 'react-icons/fi';
import { useTranslation } from 'react-i18next';
import { getAllArticles, getLocalizedArticleContent } from '../data/articles';
import Newsletter from '../components/Newsletter';

function Archive() {
  const [selectedYear, setSelectedYear] = useState('all');
  const [currentPage, setCurrentPage] = useState(1);
  const { t } = useTranslation();
  const articles = getAllArticles() || [];

  // Safely get years from articles with proper date parsing
  const years = [...new Set(articles.map(article => {
    try {
      return new Date(article.date).getFullYear();
    } catch (e) {
      return null;
    }
  }).filter(year => year !== null))].sort((a, b) => b - a);

  // Filter articles by selected year with safe date parsing
  const filteredArticles = selectedYear === 'all' 
    ? articles 
    : articles.filter(article => {
        try {
          return new Date(article.date).getFullYear() === parseInt(selectedYear);
        } catch (e) {
          return false;
        }
      });

  // Pagination logic
  const articlesPerPage = 5;
  const totalPages = Math.ceil(filteredArticles.length / articlesPerPage);
  const paginatedArticles = filteredArticles.slice((currentPage - 1) * articlesPerPage, currentPage * articlesPerPage);

  // Reset to page 1 when year changes
  React.useEffect(() => { setCurrentPage(1); }, [selectedYear]);

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-serif font-bold text-gray-900 dark:text-white mb-8">
        {t('Archive')}
      </h1>

      {/* Year Filter */}
      <div className="flex flex-wrap gap-2 mb-8">
        <button
          onClick={() => setSelectedYear('all')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            selectedYear === 'all'
              ? 'bg-green-700 text-white dark:bg-green-800'
              : 'text-gray-600 hover:bg-green-50 dark:text-gray-300 dark:hover:bg-gray-800'
          }`}
        >
          {t('All Years')}
        </button>
        {years.map(year => (
          <button
            key={year}
            onClick={() => setSelectedYear(year.toString())}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedYear === year.toString()
                ? 'bg-green-700 text-white dark:bg-green-800'
                : 'text-gray-600 hover:bg-green-50 dark:text-gray-300 dark:hover:bg-gray-800'
            }`}
          >
            {year}
          </button>
        ))}
      </div>

      {/* Articles List */}
      <div className="divide-y divide-gray-200 dark:divide-gray-700">
        {paginatedArticles.map((article) => {
          const localizedContent = getLocalizedArticleContent(article);
          // Get the appropriate color based on category
          const categoryColors = {
            analysis: 'text-blue-700 dark:text-blue-400',
            stories: 'text-amber-700 dark:text-amber-400',
            notable: 'text-purple-700 dark:text-purple-400',
            archive: 'text-green-700 dark:text-green-400'
          };
          const categoryColor = categoryColors[article.category] || categoryColors.archive;

          // Get title and content safely
          const title = localizedContent?.title || article.title;
          const content = localizedContent?.content?.[0]?.content || article.excerpt || '';

          return (
            <article key={article.id} className="py-8">
              <div className="flex items-start space-x-8">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-4 text-sm mb-2">
                    <span className={`font-medium ${categoryColor}`}>
                      {article.category.toUpperCase()}
                    </span>
                    <span className="text-gray-600 dark:text-gray-400">
                      {article.date}
                    </span>
                  </div>
                  
                  <Link to={`/article/${article.id}`}>
                    <h2 className="text-2xl font-serif font-bold text-gray-900 dark:text-white mb-3 hover:text-green-700 dark:hover:text-green-400 transition-colors">
                      {title}
                    </h2>
                  </Link>
                  
                  <p className="text-gray-600 dark:text-gray-300 mb-4 line-clamp-3 font-sans text-base">
                    {content}
                  </p>
                  
                  <div className="flex items-center gap-4 text-sm">
                    <span className="text-gray-600 dark:text-gray-400">
                      By {article.author}
                    </span>
                    <div className="flex items-center space-x-4 text-gray-500 dark:text-gray-400">
                      <span className="flex items-center">
                        <FiHeart className="mr-1 h-4 w-4" /> {article.likes}
                      </span>
                      <span className="flex items-center">
                        <FiMessageCircle className="mr-1 h-4 w-4" /> {article.comments}
                      </span>
                    </div>
                  </div>
                </div>
                
                <div className="flex-shrink-0 w-48 h-32 overflow-hidden rounded-lg">
                  <img
                    src={article.image}
                    alt={title}
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
            </article>
          );
        })}
        {filteredArticles.length === 0 && (
          <div className="text-center py-8 text-gray-600 dark:text-gray-400">
            {selectedYear === 'all' 
              ? t('No articles available yet.')
              : t('No articles available for {{year}}.', { year: selectedYear })}
          </div>
        )}
      </div>

      {/* Pagination Controls */}
      {totalPages > 1 && (
        <div className="flex justify-center mt-8 space-x-2">
          {Array.from({ length: totalPages }, (_, i) => i + 1).map(page => (
            <button
              key={page}
              onClick={() => setCurrentPage(page)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors border border-green-700 dark:border-green-800 focus:outline-none focus:ring-2 focus:ring-green-500 dark:focus:ring-green-400
                ${currentPage === page
                  ? 'bg-green-700 text-white dark:bg-green-800'
                  : 'text-green-700 bg-white hover:bg-green-50 dark:text-green-300 dark:bg-gray-900 dark:hover:bg-green-900'}
              `}
            >
              {page}
            </button>
          ))}
        </div>
      )}

      {/* Newsletter Section */}
      <div className="mt-12">
        <Newsletter variant="archive" />
      </div>
    </div>
  );
}

export default Archive; 