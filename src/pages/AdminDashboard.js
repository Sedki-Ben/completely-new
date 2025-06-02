import React, { useState } from 'react';
import ArticleEditor from '../components/ArticleEditor';
import AnalyticsDashboard from '../components/AnalyticsDashboard';
import { useAuth } from '../contexts/AuthContext';
import { articles } from '../services/api';
import { FiX, FiEdit3, FiBarChart2 } from 'react-icons/fi';

const AdminDashboard = () => {
  const { user, loading: authLoading } = useAuth();
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [activeTab, setActiveTab] = useState('editor'); // 'editor' or 'analytics'

  const handleSave = async (formData) => {
    setSaving(true);
    setError('');
    setSuccess(false);
    try {
      await articles.create(formData);
      setSuccess(true);
      // Auto-hide success message after 3 seconds
      setTimeout(() => setSuccess(false), 3000);
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
          <h1 className="text-4xl font-bold text-blue-600 dark:text-blue-400 text-center mt-2">Writer's Desk</h1>
          <p className="text-gray-600 dark:text-gray-300 text-center mt-2">Create and manage your articles</p>
          
          {/* Tabs */}
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
        </div>

        {/* Main Content */}
        <div className="relative">
          {activeTab === 'editor' ? (
            <ArticleEditor
              onSave={handleSave}
              loading={saving}
              error={error}
              userRole={user?.role}
            />
          ) : (
            <AnalyticsDashboard />
          )}
          
          {/* Toast Notification */}
          {success && (
            <div className="fixed bottom-8 right-8 bg-emerald-500 text-white px-6 py-3 rounded-lg shadow-lg flex items-center space-x-2 animate-slide-up">
              <span>Article saved successfully!</span>
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
