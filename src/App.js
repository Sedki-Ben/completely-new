import { Routes, Route, Link, useLocation } from 'react-router-dom';
import { FiSearch, FiUser } from 'react-icons/fi';
import { LanguageProvider } from './contexts/LanguageContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { useTranslation } from 'react-i18next';
import LanguageSelector from './components/LanguageSelector';
import ThemeToggle from './components/ThemeToggle';
import Home from './pages/Home';
import Analysis from './pages/Analysis';
import Stories from './pages/Stories';
import NotableWork from './pages/NotableWork';
import Archive from './pages/Archive';
import About from './pages/About';
import ArticlePage from './pages/ArticlePage';
import NotFound from './pages/NotFound';
import './App.css';

function AppContent() {
  const location = useLocation();
  const { t } = useTranslation();
  
  const isActiveRoute = (path) => {
    return location.pathname === path;
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 font-sans transition-colors duration-300">
      {/* Header - Reusable across all pages */}
      <header className="px-4 sm:px-6 py-4 bg-white dark:bg-gray-800 shadow-md sticky top-0 z-50 border-b border-gray-200 dark:border-gray-700 transition-colors duration-300">
        <div className="max-w-7xl mx-auto relative">
          {/* Title */}
          <div className="text-center mb-4 sm:mb-6">
            <Link 
              to="/" 
              className="text-4xl sm:text-5xl md:text-6xl font-serif font-bold text-gray-900 dark:text-gray-100 hover:text-gray-700 dark:hover:text-gray-300 transition-colors px-4 py-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800/50 inline-block"
            >
              Pure Tactics Cartel
            </Link>
          </div>
          
          {/* Navigation - Single line, centered */}
          <nav className="flex justify-center items-center overflow-x-auto whitespace-nowrap pb-2 sm:pb-0">
            <div className="inline-flex items-center space-x-2">
              <Link 
                to="/" 
                className={`px-3 py-2 rounded-lg font-medium text-sm transition-all duration-300 ${
                  isActiveRoute('/') 
                  ? 'bg-slate-600 dark:bg-slate-700 text-white shadow-md' 
                  : 'text-black dark:text-gray-300 hover:bg-slate-50 dark:hover:bg-gray-700'
                }`}
              >
                {t('Home')}
              </Link>
              <span className="text-gray-300 dark:text-gray-600">•</span>
              <Link 
                to="/analysis" 
                className={`px-3 py-2 rounded-lg font-medium text-sm transition-all duration-300 ${
                  isActiveRoute('/analysis') 
                  ? 'bg-blue-600 dark:bg-blue-700 text-white shadow-md' 
                  : 'text-black dark:text-gray-300 hover:bg-blue-50 dark:hover:bg-gray-700'
                }`}
              >
                {t('Analysis')}
              </Link>
              <span className="text-gray-300 dark:text-gray-600">•</span>
              <Link 
                to="/stories" 
                className={`px-3 py-2 rounded-lg font-medium text-sm transition-all duration-300 ${
                  isActiveRoute('/stories') 
                  ? 'bg-amber-600 dark:bg-amber-700 text-white shadow-md' 
                  : 'text-black dark:text-gray-300 hover:bg-amber-50 dark:hover:bg-gray-700'
                }`}
              >
                {t('Stories')}
              </Link>
              <span className="text-gray-300 dark:text-gray-600">•</span>
              <Link 
                to="/notable-work" 
                className={`px-3 py-2 rounded-lg font-medium text-sm transition-all duration-300 ${
                  isActiveRoute('/notable-work') 
                  ? 'bg-purple-600 dark:bg-purple-700 text-white shadow-md' 
                  : 'text-black dark:text-gray-300 hover:bg-purple-50 dark:hover:bg-gray-700'
                }`}
              >
                {t('Notable Work')}
              </Link>
              <span className="text-gray-300 dark:text-gray-600">•</span>
              <Link 
                to="/archive" 
                className={`px-3 py-2 rounded-lg font-medium text-sm transition-all duration-300 ${
                  isActiveRoute('/archive') 
                  ? 'bg-emerald-600 dark:bg-emerald-700 text-white shadow-md' 
                  : 'text-black dark:text-gray-300 hover:bg-emerald-50 dark:hover:bg-gray-700'
                }`}
              >
                {t('Archive')}
              </Link>
              <span className="text-gray-300 dark:text-gray-600">•</span>
              <Link 
                to="/about" 
                className={`px-3 py-2 rounded-lg font-medium text-sm transition-all duration-300 ${
                  isActiveRoute('/about') 
                  ? 'bg-gray-600 dark:bg-gray-700 text-white shadow-md' 
                  : 'text-black dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                }`}
              >
                {t('About')}
              </Link>
            </div>
          </nav>

          {/* Utility Controls */}
          <div className="flex flex-wrap items-center gap-2 justify-center sm:justify-end sm:absolute sm:top-4 sm:right-6">
            <div className="flex items-center gap-2">
              <LanguageSelector />
              <ThemeToggle />
              <button className="p-2 rounded-lg text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-all duration-300">
                <FiSearch className="w-5 h-5" />
              </button>
            </div>
            <Link 
              to="/signin" 
              className="flex items-center px-4 py-2 text-sm font-medium text-white bg-gray-700 dark:bg-gray-700 hover:bg-gray-800 dark:hover:bg-gray-600 rounded-lg transition-all duration-300 shadow-sm"
            >
              <FiUser className="w-4 h-4 mr-2" />
              {t('Sign In')}
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content - Dynamic based on route */}
      <main className="container mx-auto px-4 py-8 max-w-7xl">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route path="/stories" element={<Stories />} />
          <Route path="/notable-work" element={<NotableWork />} />
          <Route path="/archive" element={<Archive />} />
          <Route path="/about" element={<About />} />
          <Route path="/article/:id" element={<ArticlePage />} />
          <Route path="/404" element={<NotFound />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
    </div>
  );
}

function App() {
  return (
    <ThemeProvider>
      <LanguageProvider>
        <AppContent />
      </LanguageProvider>
    </ThemeProvider>
  );
}

export default App;