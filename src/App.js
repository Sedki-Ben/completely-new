import { Routes, Route, Link, useLocation } from 'react-router-dom';
import { FiSearch } from 'react-icons/fi';
import { LanguageProvider } from './contexts/LanguageContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { useTranslation } from 'react-i18next';
import LanguageSelector from './components/LanguageSelector';
import ThemeToggle from './components/ThemeToggle';
import UserAvatar from './components/UserAvatar';
import Home from './pages/Home';
import EtoileDuSahel from './pages/EtoileDuSahel';
import TheBeautifulGame from './pages/TheBeautifulGame';
import AllSportsHub from './pages/AllSportsHub';
import Archive from './pages/Archive';
import About from './pages/About';
import ArticlePage from './pages/ArticlePage';
import NotFound from './pages/NotFound';
import AdminDashboard from './pages/AdminDashboard';
import SignIn from './pages/SignIn';
import SignUp from './pages/SignUp';
import UserProfile from './pages/UserProfile';
import ProtectedRoute from './components/ProtectedRoute';
import ProtectedAdminRoute from './components/ProtectedAdminRoute';
import { AuthProvider } from './contexts/AuthContext';
import './App.css';

function AppContent() {
  const location = useLocation();
  const { t, i18n } = useTranslation();
  const isRTL = i18n.language === 'ar';
  
  const isActiveRoute = (path) => {
    return location.pathname === path;
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 font-sans transition-colors duration-300">
      {/* Header - Reusable across all pages */}
      <header className="px-4 sm:px-6 py-4 bg-white dark:bg-gray-800 shadow-md sticky top-0 z-50 border-b border-gray-200 dark:border-gray-700 transition-colors duration-300" dir="ltr">
        <div className="max-w-7xl mx-auto">
          {/* Top Bar with User Avatar and Controls */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-4" dir="ltr">
              <UserAvatar />
            </div>
            <div className="flex items-center gap-4" dir="ltr">
              <button className="p-2 rounded-lg text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-all duration-300">
                <FiSearch className="w-5 h-5" />
              </button>
              <ThemeToggle />
              <LanguageSelector />
            </div>
          </div>

          {/* Title */}
          <div className="text-center mb-4">
            <Link 
              to="/" 
              className="text-4xl sm:text-5xl md:text-6xl font-serif font-bold text-gray-900 dark:text-gray-100 hover:text-gray-700 dark:hover:text-gray-300 transition-colors px-4 py-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800/50 inline-block"
            >
              Pure Tactics Cartel
            </Link>
          </div>
          
          {/* Navigation - Single line, centered */}
          <nav className="flex justify-center items-center overflow-x-auto whitespace-nowrap pb-2 sm:pb-0" dir={isRTL ? 'rtl' : 'ltr'}>
            <div className={`inline-flex items-center ${isRTL ? 'space-x-reverse space-x-2' : 'space-x-2'}`}>
              <Link 
                to="/" 
                className={`px-3 py-2 rounded-lg font-medium text-sm transition-all duration-300 ${
                  isActiveRoute('/') 
                  ? 'bg-slate-600 dark:bg-slate-700 text-white shadow-md' 
                  : 'text-black dark:text-gray-300 hover:bg-slate-100/80 hover:text-slate-700 dark:hover:bg-slate-800/20 dark:hover:text-slate-400'
                }`}
              >
                {t('Home')}
              </Link>
              <span className="text-gray-300 dark:text-gray-600">•</span>
              <Link 
                to="/etoile-du-sahel" 
                className={`px-3 py-2 rounded-lg font-medium text-sm transition-all duration-300 ${
                  isActiveRoute('/etoile-du-sahel') 
                  ? 'bg-red-600 dark:bg-red-700 text-white shadow-md' 
                  : 'text-black dark:text-gray-300 hover:bg-red-100/80 hover:text-red-700 dark:hover:bg-red-800/20 dark:hover:text-red-400'
                }`}
              >
                {t('Etoile Du Sahel')}
              </Link>
              <span className="text-gray-300 dark:text-gray-600">•</span>
              <Link 
                to="/the-beautiful-game" 
                className={`px-3 py-2 rounded-lg font-medium text-sm transition-all duration-300 ${
                  isActiveRoute('/the-beautiful-game') 
                  ? 'bg-green-600 dark:bg-green-700 text-white shadow-md' 
                  : 'text-black dark:text-gray-300 hover:bg-green-100/80 hover:text-green-700 dark:hover:bg-green-800/20 dark:hover:text-green-400'
                }`}
              >
                {t('The Beautiful Game')}
              </Link>
              <span className="text-gray-300 dark:text-gray-600">•</span>
              <Link 
                to="/all-sports-hub" 
                className={`px-3 py-2 rounded-lg font-medium text-sm transition-all duration-300 ${
                  isActiveRoute('/all-sports-hub') 
                  ? 'bg-purple-600 dark:bg-purple-700 text-white shadow-md' 
                  : 'text-black dark:text-gray-300 hover:bg-purple-100/80 hover:text-purple-700 dark:hover:bg-purple-800/20 dark:hover:text-purple-400'
                }`}
              >
                {t('All Sports Hub')}
              </Link>
              <span className="text-gray-300 dark:text-gray-600">•</span>
              <Link 
                to="/archive" 
                className={`px-3 py-2 rounded-lg font-medium text-sm transition-all duration-300 ${
                  isActiveRoute('/archive') 
                  ? 'bg-yellow-600 dark:bg-yellow-700 text-white shadow-md' 
                  : 'text-black dark:text-gray-300 hover:bg-yellow-100/80 hover:text-yellow-700 dark:hover:bg-yellow-800/20 dark:hover:text-yellow-400'
                }`}
              >
                {t('Archive')}
              </Link>
              <span className="text-gray-300 dark:text-gray-600">•</span>
              <Link 
                to="/about" 
                className={`px-3 py-2 rounded-lg font-medium text-sm transition-all duration-300 ${
                  isActiveRoute('/about') 
                  ? 'bg-blue-600 dark:bg-blue-700 text-white shadow-md' 
                  : 'text-black dark:text-gray-300 hover:bg-blue-100/80 hover:text-blue-700 dark:hover:bg-blue-800/20 dark:hover:text-blue-400'
                }`}
              >
                {t('About')}
              </Link>
            </div>
          </nav>
        </div>
      </header>

      {/* Main Content - Dynamic based on route */}
      <main className="container mx-auto px-4 py-8 max-w-7xl">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/etoile-du-sahel" element={<EtoileDuSahel />} />
          <Route path="/the-beautiful-game" element={<TheBeautifulGame />} />
          <Route path="/all-sports-hub" element={<AllSportsHub />} />
          <Route path="/archive" element={<Archive />} />
          <Route path="/about" element={<About />} />
          <Route path="/article/:id" element={<ArticlePage />} />
          <Route path="/admin" element={<ProtectedAdminRoute><AdminDashboard /></ProtectedAdminRoute>} />
          <Route path="/admin/edit-article/:id" element={<ProtectedAdminRoute><AdminDashboard editMode={true} /></ProtectedAdminRoute>} />
          <Route path="/signin" element={<SignIn />} />
          <Route path="/signup" element={<SignUp />} />
          <Route path="/profile" element={<ProtectedRoute><UserProfile /></ProtectedRoute>} />
          <Route path="/404" element={<NotFound />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <ThemeProvider>
        <LanguageProvider>
          <AppContent />
        </LanguageProvider>
      </ThemeProvider>
    </AuthProvider>
  );
}

export default App;