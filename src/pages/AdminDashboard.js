import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import ArticleEditor from '../components/ArticleEditor';
import AnalyticsDashboard from '../components/AnalyticsDashboard';
import { useAuth } from '../contexts/AuthContext';
import { articles } from '../services/api';
import { FiX, FiEdit3, FiBarChart2, FiArrowLeft } from 'react-icons/fi';

const AdminDashboard = ({ editMode = false }) => {
  const { user, loading: authLoading } = useAuth();
  const { id } = useParams();
  const navigate = useNavigate();
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [activeTab, setActiveTab] = useState('editor'); // 'editor' or 'analytics'
  const [articleData, setArticleData] = useState(null);
  const [loadingArticle, setLoadingArticle] = useState(false);

  // Load article for editing
  useEffect(() => {
    if (editMode && id) {
      setLoadingArticle(true);
      articles.getById(id)
        .then(response => {
          const article = response.data;
          // Transform article data to match ArticleEditor expected format
          const backendUrl = process.env.REACT_APP_API_URL?.replace('/api', '') || 'http://localhost:5000';
          const fullImageUrl = article.image?.startsWith('http') ? article.image : 
                              article.image?.startsWith('/') ? `${backendUrl}${article.image}` : article.image;
          
          const transformedData = {
            ...article,
            titles: {
              en: article.translations?.en?.title || '',
              fr: article.translations?.fr?.title || '',
              ar: article.translations?.ar?.title || ''
            },
            content: {
              en: article.translations?.en?.content || [],
              fr: article.translations?.fr?.content || [],
              ar: article.translations?.ar?.content || []
            },
            image: fullImageUrl || article.image,
            type: article.category,
            tags: article.tags || []
          };
          

          setArticleData(transformedData);
        })
        .catch(err => {
          setError('Failed to load article for editing');
          console.error('Error loading article:', err);
        })
        .finally(() => {
          setLoadingArticle(false);
        });
    }
  }, [editMode, id]);

  const handleSave = async (formData) => {
    setSaving(true);
    setError('');
    setSuccess(false);
    try {
      if (editMode && id) {
        // Update existing article
        await articles.update(id, formData);
        setSuccess(true);
        // Auto-hide success message after 3 seconds
        setTimeout(() => setSuccess(false), 3000);
      } else {
        // Create new article
        await articles.create(formData);
        setSuccess(true);
        // Auto-hide success message after 3 seconds
        setTimeout(() => setSuccess(false), 3000);
      }
    } catch (err) {
      setError(err.response?.data?.message || err.response?.data?.msg || 'Failed to save article');
    } finally {
      setSaving(false);
    }
  };

  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-emerald-100 via-blue-100 to-purple-100 dark:from-gray-900 dark:via-slate-900 dark:to-gray-800">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-100 via-blue-100 to-purple-100 dark:from-gray-900 dark:via-slate-900 dark:to-gray-800 transition-colors duration-500 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="relative bg-white/80 dark:bg-slate-900/80 backdrop-blur-lg rounded-2xl shadow-2xl p-8 mb-8 border border-blue-200 dark:border-blue-900">
          {/* Accent bar */}
          <div className="absolute -top-2 left-1/2 -translate-x-1/2 w-32 h-2 rounded bg-gradient-to-r from-emerald-400 via-blue-500 to-purple-500 shadow-md" />
          {editMode && (
            <button
              onClick={() => navigate('/admin')}
              className="absolute top-4 left-4 flex items-center gap-2 px-3 py-2 bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            >
              <FiArrowLeft className="w-4 h-4" />
              Back to Dashboard
            </button>
          )}
          <h1 className="text-4xl font-bold text-blue-600 dark:text-blue-400 text-center mt-2">
            {editMode ? 'Edit Article' : "Writer's Desk"}
          </h1>
          <p className="text-gray-600 dark:text-gray-300 text-center mt-2">
            {editMode ? 'Update your published article' : 'Create and manage your articles'}
          </p>
          
          {/* Tabs - Only show in normal mode */}
          {!editMode && (
            <div className="flex justify-center mt-6 space-x-4">
              <button
                onClick={() => setActiveTab('editor')}
                className={`flex items-center px-4 py-2 rounded-lg transition-colors ${
                  activeTab === 'editor'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                }`}
              >
                <FiEdit3 className="w-5 h-5 mr-2" />
                Editor
              </button>
              <button
                onClick={() => setActiveTab('analytics')}
                className={`flex items-center px-4 py-2 rounded-lg transition-colors ${
                  activeTab === 'analytics'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                }`}
              >
                <FiBarChart2 className="w-5 h-5 mr-2" />
                Analytics
              </button>
            </div>
          )}
        </div>

        {/* Main Content */}
        <div className="relative">
          {editMode ? (
            loadingArticle ? (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent"></div>
                <span className="ml-3 text-gray-600 dark:text-gray-400">Loading article...</span>
              </div>
            ) : articleData ? (
              <ArticleEditor
                onSave={handleSave}
                loading={saving}
                error={error}
                userRole={user?.role}
                initialData={articleData}
              />
            ) : (
              <div className="flex items-center justify-center py-12">
                <span className="text-gray-600 dark:text-gray-400">No article data available</span>
              </div>
            )
          ) : (
            activeTab === 'editor' ? (
              <ArticleEditor
                onSave={handleSave}
                loading={saving}
                error={error}
                userRole={user?.role}
              />
            ) : (
              <AnalyticsDashboard />
            )
          )}
          
          {/* Toast Notification */}
          {success && (
            <div className="fixed bottom-8 right-8 bg-emerald-500 text-white px-6 py-3 rounded-lg shadow-lg flex items-center space-x-2 animate-slide-up">
              <span>{editMode ? 'Article updated successfully!' : 'Article saved successfully!'}</span>
              <button 
                onClick={() => setSuccess(false)}
                className="text-white hover:text-emerald-100 transition-colors"
              >
                <FiX className="w-5 h-5" />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
