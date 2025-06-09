import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { FiArrowRight, FiStar } from 'react-icons/fi';
import { GiBoxingGlove } from 'react-icons/gi';
import { useTranslation } from 'react-i18next';
import { useArticles, getLocalizedArticleContent } from '../hooks/useArticles';
import Newsletter from '../components/Newsletter';

function Home() {
  const { t, i18n } = useTranslation();
  const { fetchArticlesByCategory, loading, error } = useArticles();
  
  const [etoileArticles, setEtoileArticles] = useState([]);
  const [beautifulGameArticles, setBeautifulGameArticles] = useState([]);
  const [allSportsArticles, setAllSportsArticles] = useState([]);

  useEffect(() => {
    const loadArticles = async () => {
      try {
        // First try to use cached articles for each category
        const [cachedEtoile, cachedBeautiful, cachedSports] = await Promise.all([
          fetchArticlesByCategory('etoile-du-sahel', true),
          fetchArticlesByCategory('the-beautiful-game', true),
          fetchArticlesByCategory('all-sports-hub', true)
        ]);
        
        // Set initial articles from cache if available
        if (cachedEtoile && cachedEtoile.length > 0) setEtoileArticles(cachedEtoile);
        if (cachedBeautiful && cachedBeautiful.length > 0) setBeautifulGameArticles(cachedBeautiful);
        if (cachedSports && cachedSports.length > 0) setAllSportsArticles(cachedSports);
        
        // Fetch fresh articles for categories that have insufficient cache
        const refreshPromises = [];
        
        if (!cachedEtoile || cachedEtoile.length < 3) {
          refreshPromises.push(
            fetchArticlesByCategory('etoile-du-sahel', false).then(articles => {
              if (articles) setEtoileArticles(articles);
              return articles;
            })
          );
        }
        
        if (!cachedBeautiful || cachedBeautiful.length < 3) {
          refreshPromises.push(
            fetchArticlesByCategory('the-beautiful-game', false).then(articles => {
              if (articles) setBeautifulGameArticles(articles);
              return articles;
            })
          );
        }
        
        if (!cachedSports || cachedSports.length < 3) {
          refreshPromises.push(
            fetchArticlesByCategory('all-sports-hub', false).then(articles => {
              if (articles) setAllSportsArticles(articles);
              return articles;
            })
          );
        }
        
        // Wait for any necessary refreshes
        if (refreshPromises.length > 0) {
          await Promise.all(refreshPromises);
        }
      } catch (error) {
        console.error('Error loading articles:', error);
      }
    };

    loadArticles();
  }, [fetchArticlesByCategory]);

  const featuredEtoile = etoileArticles[0];
  const recentEtoile = etoileArticles.slice(1, 5);
  
  const featuredGame = beautifulGameArticles[0];
  const recentGames = beautifulGameArticles.slice(1, 5);

  const featuredSports = allSportsArticles[0];
  const recentSports = allSportsArticles.slice(1, 5);

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
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-gray-900 mx-auto"></div>
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
            className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            {t('Retry')}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-stone-50 to-gray-100 dark:from-gray-900 dark:via-slate-900/20 dark:to-gray-900">
      <div className="container mx-auto px-4 py-8">
        
        {/* Hero Section */}
        <div className="text-center mb-20 relative">
          <div className="absolute inset-0 bg-slate-500/5 dark:bg-slate-400/10 rounded-full blur-3xl transform -translate-y-12"></div>
          <h1 className="text-7xl font-serif font-bold bg-gradient-to-r from-slate-600 via-gray-700 to-slate-800 dark:from-slate-400 dark:via-gray-300 dark:to-slate-200 bg-clip-text text-transparent mb-6 relative">
            {t('Home')}
          </h1>
          <div className="w-24 h-1 bg-gradient-to-r from-slate-500 to-slate-700 mx-auto rounded-full mb-8"></div>
          <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto leading-relaxed">
            {t('Welcome Text')}
          </p>
        </div>

        {/* Etoile Du Sahel Section */}
        <section className="mb-20">
          <div className="max-w-7xl mx-auto">
            <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-lg rounded-3xl p-8 shadow-2xl border border-red-100 dark:border-red-800/30 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-red-500/20 to-transparent rounded-full transform translate-x-16 -translate-y-16"></div>
              <div className="absolute bottom-0 left-0 w-24 h-24 bg-gradient-to-tr from-red-600/10 to-transparent rounded-full transform -translate-x-12 translate-y-12"></div>
              
              <div className="relative">
                <div className="flex justify-between items-center mb-10">
                  <Link to="/etoile-du-sahel" className="flex items-center gap-4 group">
                    <div className="w-16 h-16 bg-gradient-to-br from-red-500 to-red-700 rounded-2xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-300">
                      <FiStar className="text-2xl text-white" />
                    </div>
                    <h2 className={`text-4xl font-serif font-bold ${themeColors['etoile-du-sahel'].text} group-hover:scale-105 transition-transform duration-300`}>
                      {t('Etoile Du Sahel')}
                    </h2>
                  </Link>
                  <Link 
                    to="/etoile-du-sahel" 
                    className={`flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 font-medium`}
                  >
                    {t('View All')} <FiArrowRight className="w-4 h-4" />
                  </Link>
                </div>
                
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                  {/* Featured Etoile Article */}
                  {featuredEtoile && (
                    <div className="lg:col-span-7">
                      <Link to={`/article/${featuredEtoile.slug || featuredEtoile.id}`} className="group">
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
                        to={`/article/${article.slug || article.id}`}
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
              </div>
            </div>
          </div>
        </section>

        {/* The Beautiful Game Section */}
        <section className="mb-20">
          <div className="max-w-7xl mx-auto">
            <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-lg rounded-3xl p-8 shadow-2xl border border-green-100 dark:border-green-800/30 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-green-500/20 to-transparent rounded-full transform translate-x-16 -translate-y-16"></div>
              <div className="absolute bottom-0 left-0 w-24 h-24 bg-gradient-to-tr from-emerald-600/10 to-transparent rounded-full transform -translate-x-12 translate-y-12"></div>
              
              <div className="relative">
                <div className="flex justify-between items-center mb-10">
                  <Link to="/the-beautiful-game" className="flex items-center gap-4 group">
                    <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-emerald-700 rounded-2xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-300">
                      <span className="text-2xl">⚽</span>
                    </div>
                    <h2 className={`text-4xl font-serif font-bold ${themeColors['the-beautiful-game'].text} group-hover:scale-105 transition-transform duration-300`}>
                      {t('The Beautiful Game')}
                    </h2>
                  </Link>
                  <Link 
                    to="/the-beautiful-game" 
                    className={`flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 font-medium`}
                  >
                    {t('View All')} <FiArrowRight className="w-4 h-4" />
                  </Link>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                  {/* Featured Game Article */}
                  {featuredGame && (
                    <div className="lg:col-span-7">
                      <Link to={`/article/${featuredGame.slug || featuredGame.id}`} className="group">
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
                        to={`/article/${article.slug || article.id}`}
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
              </div>
            </div>
          </div>
        </section>

        {/* All-Sports Hub Section */}
        <section className="mb-20">
          <div className="max-w-7xl mx-auto">
            <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-lg rounded-3xl p-8 shadow-2xl border border-purple-100 dark:border-purple-800/30 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-purple-500/20 to-transparent rounded-full transform translate-x-16 -translate-y-16"></div>
              <div className="absolute bottom-0 left-0 w-24 h-24 bg-gradient-to-tr from-indigo-600/10 to-transparent rounded-full transform -translate-x-12 translate-y-12"></div>
              
              <div className="relative">
                <div className="flex justify-between items-center mb-10">
                  <Link to="/all-sports-hub" className="flex items-center gap-4 group">
                    <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-indigo-700 rounded-2xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-300">
                      <GiBoxingGlove className="text-2xl text-white" />
                    </div>
                    <h2 className={`text-4xl font-serif font-bold ${themeColors['all-sports-hub'].text} group-hover:scale-105 transition-transform duration-300`}>
                      {t('All-Sports Hub')}
                    </h2>
                  </Link>
                  <Link 
                    to="/all-sports-hub" 
                    className={`flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-500 to-indigo-600 hover:from-purple-600 hover:to-indigo-700 text-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 font-medium`}
                  >
                    {t('View All')} <FiArrowRight className="w-4 h-4" />
                  </Link>
                </div>
                
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                  {/* Featured Sports Article */}
                  {featuredSports && (
                    <div className="lg:col-span-7">
                      <Link to={`/article/${featuredSports.slug || featuredSports.id}`} className="group">
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
                        to={`/article/${article.slug || article.id}`}
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
              </div>
            </div>
          </div>
        </section>

        {/* Newsletter Section */}
        <section className="mb-20">
          <div className="max-w-7xl mx-auto">
            <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-lg rounded-3xl p-8 shadow-2xl border border-slate-100 dark:border-slate-800/30 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-slate-500/20 to-transparent rounded-full transform translate-x-16 -translate-y-16"></div>
              <div className="absolute bottom-0 left-0 w-24 h-24 bg-gradient-to-tr from-gray-600/10 to-transparent rounded-full transform -translate-x-12 translate-y-12"></div>
              
              <div className="relative">
                <Newsletter variant="default" />
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

export default Home; 