import React, { useState } from 'react';
import ArticleEditor from '../components/ArticleEditor';
import { useAuth } from '../contexts/AuthContext';
import { articles } from '../services/api';

const AdminDashboard = () => {
  const { user, loading: authLoading } = useAuth();
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  const handleSave = async (formData) => {
    setSaving(true);
    setError('');
    setSuccess(false);
    try {
      await articles.create(formData);
      setSuccess(true);
    } catch (err) {
      setError(err.response?.data?.message || err.response?.data?.msg || 'Failed to save article');
    } finally {
      setSaving(false);
    }
  };

  if (authLoading) return <div>Loading...</div>;

  return (
    <div className="min-h-screen py-10 px-4 bg-slate-50 dark:bg-slate-900">
      <h1 className="text-3xl font-bold mb-8">Admin Dashboard</h1>
      <div className="mb-10">
        <ArticleEditor
          onSave={handleSave}
          loading={saving}
          error={error}
          userRole={user?.role}
        />
        {success && <div className="text-green-600 mt-4">Article saved successfully!</div>}
      </div>
      {/* Future dashboard features go here */}
    </div>
  );
};

export default AdminDashboard;
