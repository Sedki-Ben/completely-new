import React from 'react';
import { Link } from 'react-router-dom';
import { FiHeart, FiMessageCircle, FiShare2, FiArrowRight } from 'react-icons/fi';
import { useTranslation } from 'react-i18next';
import { getAllArticles, getArticlesByCategory, getLocalizedArticleContent, categoryTranslations } from '../data/articles';
import Newsletter from '../components/Newsletter';

function Home() {
  const { t, i18n } = useTranslation();
  const etoileArticles = getArticlesByCategory('etoile-du-sahel');
  const beautifulGameArticles = getArticlesByCategory('the-beautiful-game');
  const allSportsArticles = getArticlesByCategory('all-sports-hub');
  const archiveArticles = getArticlesByCategory('archive');

  const featuredEtoile = etoileArticles[0];
  const recentEtoile = etoileArticles.slice(1, 5);
  
  const featuredGame = beautifulGameArticles[0];
  const recentGames = beautifulGameArticles.slice(1, 5);

  const featuredSports = allSportsArticles[0];
  const recentSports = allSportsArticles.slice(1, 5);

  const featuredArchive = archiveArticles[0];
  const recentArchive = archiveArticles.slice(1, 5);

  const themeColors = {
    'etoile-du-sahel': {
      text: 'text-red-900 dark:text-red-400',
      hover: 'hover:text-red-700 dark:hover:text-red-300',
      groupHover: 'group-hover:text-red-900 dark:group-hover:text-red-400'
    },
    'the-beautiful-game': {
      text: 'text-green-900 dark:text-green-400',
      hover: 'hover:text-green-700 dark:hover:text-green-300',
      groupHover: 'group-hover:text-green-900 dark:group-hover:text-green-400'
    },
    'all-sports-hub': {
      text: 'text-purple-900 dark:text-purple-400',
      hover: 'hover:text-purple-700 dark:hover:text-purple-300',
      groupHover: 'group-hover:text-purple-900 dark:group-hover:text-purple-400'
    },
    'archive': {
      text: 'text-yellow-900 dark:text-yellow-400',
      hover: 'hover:text-yellow-700 dark:hover:text-yellow-300',
      groupHover: 'group-hover:text-yellow-900 dark:group-hover:text-yellow-400'
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="container mx-auto px-4 py-8">
        {/* Etoile Du Sahel Section */}
        <section className="mb-16">
          <div className="flex justify-between items-center mb-8">
            <h2 className={`text-4xl font-serif font-bold ${themeColors['etoile-du-sahel'].text}`}>
              {t('Etoile Du Sahel')}
            </h2>
            <Link 
              to="/etoile-du-sahel" 
              className={`flex items-center ${themeColors['etoile-du-sahel'].text} ${themeColors['etoile-du-sahel'].hover} transition-colors`}
            >
              {t('View All')} <FiArrowRight className="ml-2" />
            </Link>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Featured Etoile Article */}
            {featuredEtoile && (
              <div className="lg:col-span-7">
                <Link to={`/article/${featuredEtoile.id}`} className="group">
                  <div className="aspect-w-16 aspect-h-9 mb-4">
                    <img
                      src={featuredEtoile.image}
                      alt={getLocalizedArticleContent(featuredEtoile, i18n.language)?.title}
                      className="w-full h-full object-cover rounded-lg"
                    />
                  </div>
                  <h3 className={`text-2xl font-serif font-bold text-gray-900 dark:text-white mb-2 ${themeColors['etoile-du-sahel'].groupHover} transition-colors`}>
                    {getLocalizedArticleContent(featuredEtoile, i18n.language)?.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300 mb-3 line-clamp-2">
                    {getLocalizedArticleContent(featuredEtoile, i18n.language)?.excerpt}
                  </p>
                  <div className="text-sm text-gray-500 dark:text-gray-400">
                    {(i18n.language === 'ar' ? 'صدقي بن حوالة' : featuredEtoile.author)} • {featuredEtoile.date}
                  </div>
                </Link>
              </div>
            )}

            {/* Recent Etoile Articles List */}
            <div className="lg:col-span-5 space-y-6">
              {recentEtoile.map(article => (
                <Link 
                  key={article.id}
                  to={`/article/${article.id}`}
                  className="flex gap-4 group"
                >
                  <div className="flex-grow">
                    <h4 className={`font-serif font-bold text-gray-900 dark:text-white mb-2 ${themeColors['etoile-du-sahel'].groupHover} transition-colors`}>
                      {getLocalizedArticleContent(article, i18n.language)?.title}
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-300 mb-2 line-clamp-2">
                      {getLocalizedArticleContent(article, i18n.language)?.excerpt}
                    </p>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      {(i18n.language === 'ar' ? 'صدقي بن حوالة' : article.author)} • {article.date}
                    </div>
                  </div>
                  <div className="flex-shrink-0 w-24 h-24">
                    <img
                      src={article.image}
                      alt={getLocalizedArticleContent(article, i18n.language)?.title}
                      className="w-full h-full object-cover rounded-lg"
                    />
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </section>

        {/* The Beautiful Game Section */}
        <section className="mb-16">
          <div className="flex justify-between items-center mb-8">
            <h2 className={`text-4xl font-serif font-bold ${themeColors['the-beautiful-game'].text}`}>
              {t('The Beautiful Game')}
            </h2>
            <Link 
              to="/the-beautiful-game" 
              className={`flex items-center ${themeColors['the-beautiful-game'].text} ${themeColors['the-beautiful-game'].hover} transition-colors`}
            >
              {t('View All')} <FiArrowRight className="ml-2" />
            </Link>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Featured Game Article */}
            {featuredGame && (
              <div className="lg:col-span-7">
                <Link to={`/article/${featuredGame.id}`} className="group">
                  <div className="aspect-w-16 aspect-h-9 mb-4">
                    <img
                      src={featuredGame.image}
                      alt={getLocalizedArticleContent(featuredGame, i18n.language)?.title}
                      className="w-full h-full object-cover rounded-lg"
                    />
                  </div>
                  <h3 className={`text-2xl font-serif font-bold text-gray-900 dark:text-white mb-2 ${themeColors['the-beautiful-game'].groupHover} transition-colors`}>
                    {getLocalizedArticleContent(featuredGame, i18n.language)?.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300 mb-3 line-clamp-2">
                    {getLocalizedArticleContent(featuredGame, i18n.language)?.excerpt}
                  </p>
                  <div className="text-sm text-gray-500 dark:text-gray-400">
                    {(i18n.language === 'ar' ? 'صدقي بن حوالة' : featuredGame.author)} • {featuredGame.date}
                  </div>
                </Link>
              </div>
            )}

            {/* Recent Game Articles List */}
            <div className="lg:col-span-5 space-y-6">
              {recentGames.map(article => (
                <Link 
                  key={article.id}
                  to={`/article/${article.id}`}
                  className="flex gap-4 group"
                >
                  <div className="flex-grow">
                    <h4 className={`font-serif font-bold text-gray-900 dark:text-white mb-2 ${themeColors['the-beautiful-game'].groupHover} transition-colors`}>
                      {getLocalizedArticleContent(article, i18n.language)?.title}
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-300 mb-2 line-clamp-2">
                      {getLocalizedArticleContent(article, i18n.language)?.excerpt}
                    </p>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      {(i18n.language === 'ar' ? 'صدقي بن حوالة' : article.author)} • {article.date}
                    </div>
                  </div>
                  <div className="flex-shrink-0 w-24 h-24">
                    <img
                      src={article.image}
                      alt={getLocalizedArticleContent(article, i18n.language)?.title}
                      className="w-full h-full object-cover rounded-lg"
                    />
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </section>

        {/* All-Sports Hub Section */}
        <section className="mb-16">
          <div className="flex justify-between items-center mb-8">
            <h2 className={`text-4xl font-serif font-bold ${themeColors['all-sports-hub'].text}`}>
              {t('All-Sports Hub')}
            </h2>
            <Link 
              to="/all-sports-hub" 
              className={`flex items-center ${themeColors['all-sports-hub'].text} ${themeColors['all-sports-hub'].hover} transition-colors`}
            >
              {t('View All')} <FiArrowRight className="ml-2" />
            </Link>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Featured Sports Article */}
            {featuredSports && (
              <div className="lg:col-span-7">
                <Link to={`/article/${featuredSports.id}`} className="group">
                  <div className="aspect-w-16 aspect-h-9 mb-4">
                    <img
                      src={featuredSports.image}
                      alt={getLocalizedArticleContent(featuredSports, i18n.language)?.title}
                      className="w-full h-full object-cover rounded-lg"
                    />
                  </div>
                  <h3 className={`text-2xl font-serif font-bold text-gray-900 dark:text-white mb-2 ${themeColors['all-sports-hub'].groupHover} transition-colors`}>
                    {getLocalizedArticleContent(featuredSports, i18n.language)?.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300 mb-3 line-clamp-2">
                    {getLocalizedArticleContent(featuredSports, i18n.language)?.excerpt}
                  </p>
                  <div className="text-sm text-gray-500 dark:text-gray-400">
                    {(i18n.language === 'ar' ? 'صدقي بن حوالة' : featuredSports.author)} • {featuredSports.date}
                  </div>
                </Link>
              </div>
            )}

            {/* Recent Sports Articles List */}
            <div className="lg:col-span-5 space-y-6">
              {recentSports.map(article => (
                <Link 
                  key={article.id}
                  to={`/article/${article.id}`}
                  className="flex gap-4 group"
                >
                  <div className="flex-grow">
                    <h4 className={`font-serif font-bold text-gray-900 dark:text-white mb-2 ${themeColors['all-sports-hub'].groupHover} transition-colors`}>
                      {getLocalizedArticleContent(article, i18n.language)?.title}
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-300 mb-2 line-clamp-2">
                      {getLocalizedArticleContent(article, i18n.language)?.excerpt}
                    </p>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      {(i18n.language === 'ar' ? 'صدقي بن حوالة' : article.author)} • {article.date}
                    </div>
                  </div>
                  <div className="flex-shrink-0 w-24 h-24">
                    <img
                      src={article.image}
                      alt={getLocalizedArticleContent(article, i18n.language)?.title}
                      className="w-full h-full object-cover rounded-lg"
                    />
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </section>

        {/* Archive Section */}
        <section className="mb-16">
          <div className="flex justify-between items-center mb-8">
            <h2 className={`text-4xl font-serif font-bold ${themeColors['archive'].text}`}>
              {t('Archive')}
            </h2>
            <Link 
              to="/archive" 
              className={`flex items-center ${themeColors['archive'].text} ${themeColors['archive'].hover} transition-colors`}
            >
              {t('View All')} <FiArrowRight className="ml-2" />
            </Link>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Featured Archive Article */}
            {featuredArchive && (
              <div className="lg:col-span-7">
                <Link to={`/article/${featuredArchive.id}`} className="group">
                  <div className="aspect-w-16 aspect-h-9 mb-4">
                    <img
                      src={featuredArchive.image}
                      alt={getLocalizedArticleContent(featuredArchive, i18n.language)?.title}
                      className="w-full h-full object-cover rounded-lg"
                    />
                  </div>
                  <h3 className={`text-2xl font-serif font-bold text-gray-900 dark:text-white mb-2 ${themeColors['archive'].groupHover} transition-colors`}>
                    {getLocalizedArticleContent(featuredArchive, i18n.language)?.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300 mb-3 line-clamp-2">
                    {getLocalizedArticleContent(featuredArchive, i18n.language)?.excerpt}
                  </p>
                  <div className="text-sm text-gray-500 dark:text-gray-400">
                    {(i18n.language === 'ar' ? 'صدقي بن حوالة' : featuredArchive.author)} • {featuredArchive.date}
                  </div>
                </Link>
              </div>
            )}

            {/* Recent Archive Articles List */}
            <div className="lg:col-span-5 space-y-6">
              {recentArchive.map(article => (
                <Link 
                  key={article.id}
                  to={`/article/${article.id}`}
                  className="flex gap-4 group"
                >
                  <div className="flex-grow">
                    <h4 className={`font-serif font-bold text-gray-900 dark:text-white mb-2 ${themeColors['archive'].groupHover} transition-colors`}>
                      {getLocalizedArticleContent(article, i18n.language)?.title}
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-300 mb-2 line-clamp-2">
                      {getLocalizedArticleContent(article, i18n.language)?.excerpt}
                    </p>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      {(i18n.language === 'ar' ? 'صدقي بن حوالة' : article.author)} • {article.date}
                    </div>
                  </div>
                  <div className="flex-shrink-0 w-24 h-24">
                    <img
                      src={article.image}
                      alt={getLocalizedArticleContent(article, i18n.language)?.title}
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