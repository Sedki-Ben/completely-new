import React from 'react';
import { Link } from 'react-router-dom';
import { FiHeart, FiMessageCircle, FiShare2, FiArrowRight } from 'react-icons/fi';
import { useTranslation } from 'react-i18next';
import { getAllArticles, getArticlesByCategory, getLocalizedArticleContent, categoryTranslations } from '../data/articles';
import Newsletter from '../components/Newsletter';

function Home() {
  const { t, i18n } = useTranslation();
  const analysisArticles = getArticlesByCategory('analysis');
  const storiesArticles = getArticlesByCategory('stories');

  const featuredAnalysis = analysisArticles[0];
  const recentAnalysis = analysisArticles.slice(1, 5);
  
  const featuredStory = storiesArticles[0];
  const recentStories = storiesArticles.slice(1, 5);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="container mx-auto px-4 py-8">
        {/* Analysis Section */}
        <section className="mb-16">
          <div className="flex justify-between items-center mb-8">
            <h2 className="text-4xl font-serif font-bold text-gray-900 dark:text-white">
              {t('Analysis')}
            </h2>
            <Link 
              to="/analysis" 
              className="flex items-center text-blue-900 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-colors"
            >
              {t('Analysis')} <FiArrowRight className="ml-2" />
            </Link>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Featured Analysis */}
            <div className="lg:col-span-7">
              <Link to={`/article/${featuredAnalysis.id}`} className="group">
                <div className="aspect-w-16 aspect-h-9 mb-4">
                  <img
                    src={featuredAnalysis.image}
                    alt={getLocalizedArticleContent(featuredAnalysis, i18n.language).title}
                    className="w-full h-full object-cover rounded-lg"
                  />
                </div>
                <h3 className="text-2xl font-serif font-bold text-gray-900 dark:text-white mb-2 group-hover:text-blue-900 dark:group-hover:text-blue-400 transition-colors">
                  {getLocalizedArticleContent(featuredAnalysis, i18n.language).title}
                </h3>
                <p className="text-gray-600 dark:text-gray-300 mb-3 line-clamp-2">
                  {getLocalizedArticleContent(featuredAnalysis, i18n.language).excerpt}
                </p>
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  {(i18n.language === 'ar' ? 'صدقي بن حوالة' : featuredAnalysis.author)} • {featuredAnalysis.date}
                </div>
              </Link>
            </div>

            {/* Recent Analysis List */}
            <div className="lg:col-span-5 space-y-6">
              {recentAnalysis.map(article => (
                <Link 
                  key={article.id}
                  to={`/article/${article.id}`}
                  className="flex gap-4 group"
                >
                  <div className="flex-grow">
                    <h4 className="font-serif font-bold text-gray-900 dark:text-white mb-2 group-hover:text-blue-900 dark:group-hover:text-blue-400 transition-colors">
                      {getLocalizedArticleContent(article, i18n.language).title}
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-300 mb-2 line-clamp-2">
                      {getLocalizedArticleContent(article, i18n.language).excerpt}
                    </p>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      {(i18n.language === 'ar' ? 'صدقي بن حوالة' : article.author)} • {article.date}
                    </div>
                  </div>
                  <div className="flex-shrink-0 w-24 h-24">
                    <img
                      src={article.image}
                      alt={getLocalizedArticleContent(article, i18n.language).title}
                      className="w-full h-full object-cover rounded-lg"
                    />
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </section>

        {/* Stories Section */}
        <section className="mb-16">
          <div className="flex justify-between items-center mb-8">
            <h2 className="text-4xl font-serif font-bold text-gray-900 dark:text-white">
              {categoryTranslations.stories[i18n.language]}
            </h2>
            <Link 
              to="/stories" 
              className="flex items-center text-blue-900 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-colors"
            >
              {t('View All')} <FiArrowRight className="ml-2" />
            </Link>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Featured Story */}
            <div className="lg:col-span-7">
              <Link to={`/article/${featuredStory.id}`} className="group">
                <div className="aspect-w-16 aspect-h-9 mb-4">
                  <img
                    src={featuredStory.image}
                    alt={getLocalizedArticleContent(featuredStory, i18n.language).title}
                    className="w-full h-full object-cover rounded-lg"
                  />
                </div>
                <h3 className="text-2xl font-serif font-bold text-gray-900 dark:text-white mb-2 group-hover:text-blue-900 dark:group-hover:text-blue-400 transition-colors">
                  {getLocalizedArticleContent(featuredStory, i18n.language).title}
                </h3>
                <p className="text-gray-600 dark:text-gray-300 mb-3 line-clamp-2">
                  {getLocalizedArticleContent(featuredStory, i18n.language).excerpt}
                </p>
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  {(i18n.language === 'ar' ? 'صدقي بن حوالة' : featuredStory.author)} • {featuredStory.date}
                </div>
              </Link>
            </div>

            {/* Recent Stories List */}
            <div className="lg:col-span-5 space-y-6">
              {recentStories.map(article => (
                <Link 
                  key={article.id}
                  to={`/article/${article.id}`}
                  className="flex gap-4 group"
                >
                  <div className="flex-grow">
                    <h4 className="font-serif font-bold text-gray-900 dark:text-white mb-2 group-hover:text-blue-900 dark:group-hover:text-blue-400 transition-colors">
                      {getLocalizedArticleContent(article, i18n.language).title}
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-300 mb-2 line-clamp-2">
                      {getLocalizedArticleContent(article, i18n.language).excerpt}
                    </p>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      {(i18n.language === 'ar' ? 'صدقي بن حوالة' : article.author)} • {article.date}
                    </div>
                  </div>
                  <div className="flex-shrink-0 w-24 h-24">
                    <img
                      src={article.image}
                      alt={getLocalizedArticleContent(article, i18n.language).title}
                      className="w-full h-full object-cover rounded-lg"
                    />
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </section>

        {/* Newsletter Section */}
        <section className="mb-16">
          <div className="max-w-4xl mx-auto">
            <Newsletter variant="default" />
          </div>
        </section>
      </div>
    </div>
  );
}

export default Home; 