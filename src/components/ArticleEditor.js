import React, { useState } from 'react';
import ReactQuill from 'react-quill';
import 'react-quill/dist/quill.snow.css';

const categories = [
  'Analysis',
  'Stories',
  'Notable Work',
  'Archive',
];

const ArticleEditor = ({ onSave, onCancel, initialData = {}, loading = false, error = '', userRole }) => {
  // Move all hooks to the top
  const [title, setTitle] = useState(initialData.title || '');
  const [content, setContent] = useState(initialData.content || '');
  const [tags, setTags] = useState(initialData.tags ? initialData.tags.join(', ') : '');
  const [category, setCategory] = useState(initialData.category || categories[0]);
  const [image, setImage] = useState(null);
  const [localError, setLocalError] = useState('');

  // Permission check
  const hasPermission = ['writer', 'admin'].includes(userRole);

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setImage(e.target.files[0]);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!title || !content) {
      setLocalError('Title and content are required.');
      return;
    }
    setLocalError('');
    const formData = new FormData();
    formData.append('title', title);
    formData.append('content', content);
    formData.append('tags', tags.split(',').map(t => t.trim()).filter(Boolean));
    formData.append('category', category);
    if (image) formData.append('image', image);
    onSave(formData);
  };

  if (!hasPermission) {
    return <div className="text-red-500">You do not have permission to access this page.</div>;
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6 max-w-2xl mx-auto bg-white dark:bg-slate-900 p-8 rounded-xl shadow-lg">
      <h2 className="text-2xl font-bold mb-4">{initialData._id ? 'Edit Article' : 'New Article'}</h2>
      <input
        className="border p-2 w-full rounded"
        value={title}
        onChange={e => setTitle(e.target.value)}
        placeholder="Title"
        required
        disabled={loading}
      />
      <ReactQuill
        value={content}
        onChange={setContent}
        className="bg-white dark:bg-slate-800 rounded"
        theme="snow"
        readOnly={loading}
      />
      <input
        className="border p-2 w-full rounded"
        value={tags}
        onChange={e => setTags(e.target.value)}
        placeholder="Tags (comma separated)"
        disabled={loading}
      />
      <select
        className="border p-2 w-full rounded"
        value={category}
        onChange={e => setCategory(e.target.value)}
        disabled={loading}
      >
        {categories.map(cat => (
          <option key={cat} value={cat}>{cat}</option>
        ))}
      </select>
      <input
        type="file"
        accept="image/*"
        className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        onChange={handleImageChange}
        disabled={loading}
      />
      {(image || initialData.coverImage) && (
        <div className="mt-4">
          <img
            src={image ? URL.createObjectURL(image) : initialData.coverImage}
            alt="Preview"
            className="max-h-48 rounded shadow border"
          />
        </div>
      )}
      {(localError || error) && <div className="text-red-500">{localError || error}</div>}
      <div className="flex gap-4 mt-4">
        <button type="submit" className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 disabled:opacity-50" disabled={loading}>
          {loading ? 'Saving...' : 'Save Article'}
        </button>
        {onCancel && (
          <button type="button" className="bg-gray-300 text-gray-800 px-6 py-2 rounded hover:bg-gray-400" onClick={onCancel} disabled={loading}>
            Cancel
          </button>
        )}
      </div>
    </form>
  );
};

export default ArticleEditor;