import React, { useState, useCallback, useRef } from 'react';
import ReactQuill from 'react-quill';
import 'react-quill/dist/quill.snow.css';
import { FiImage, FiX, FiEye } from 'react-icons/fi';

const types = [
  'etoile-du-sahel',
  'the-beautiful-game',
  'all-sports-hub'
];

const ArticleEditor = ({ onSave, onCancel, initialData = {}, loading = false, error = '', userRole }) => {
  // Move all hooks to the top
  const [title, setTitle] = useState(initialData.title || '');
  const [content, setContent] = useState(initialData.content || '');
  const [tags, setTags] = useState(initialData.tags ? initialData.tags.join(', ') : '');
  const [type, setType] = useState(initialData.type || types[0]);
  const [images, setImages] = useState([]);
  const [localError, setLocalError] = useState('');
  const [showPreview, setShowPreview] = useState(false);
  const [dragging, setDragging] = useState(false);

  // Permission check
  const hasPermission = ['writer', 'admin'].includes(userRole);

  const handleImageChange = useCallback((e) => {
    const files = Array.from(e.target.files);
    const validFiles = files.filter(file => {
      const isValid = file.type.startsWith('image/');
      const isValidSize = file.size <= 5 * 1024 * 1024; // 5MB limit
      if (!isValid) setLocalError('Please upload only image files.');
      if (!isValidSize) setLocalError('Image size should be less than 5MB.');
      return isValid && isValidSize;
    });

    setImages(prev => [...prev, ...validFiles.map(file => ({
      file,
      preview: URL.createObjectURL(file),
      position: content.length // Default to end of content
    }))]);
  }, []);

  const removeImage = (index) => {
    setImages(prev => {
      const newImages = [...prev];
      URL.revokeObjectURL(newImages[index].preview);
      newImages.splice(index, 1);
      return newImages;
    });
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
    formData.append('type', type);
    
    const tagsArray = tags.split(',').map(t => t.trim()).filter(Boolean);
    tagsArray.forEach((tag, index) => {
      formData.append(`tags[${index}]`, tag);
    });
    
    images.forEach((img, index) => {
      formData.append(`images[${index}]`, img.file);
      formData.append(`imagePositions[${index}]`, img.position);
    });
    
    onSave(formData);
  };

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    setImages(prevImages => [...prevImages, ...files]);
    setDragging(false);
  }, []);

  if (!hasPermission) {
    return (
      <div className="bg-red-100 dark:bg-red-900/50 text-red-600 dark:text-red-300 p-4 rounded-lg text-center">
        You do not have permission to access this page.
      </div>
    );
  }

  return (
    <div className="bg-white/80 dark:bg-slate-900/80 backdrop-blur-lg rounded-2xl shadow-2xl p-8 border border-blue-200 dark:border-blue-900">
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Title Input */}
      <input
          className="w-full px-4 py-3 rounded-lg border border-blue-300 dark:border-blue-700 bg-white/70 dark:bg-slate-800/70 focus:outline-none focus:ring-2 focus:ring-blue-400 dark:focus:ring-blue-600 transition placeholder-gray-400 dark:placeholder-gray-500 text-blue-900 dark:text-blue-100"
        value={title}
        onChange={e => setTitle(e.target.value)}
          placeholder="Article Title"
        required
        disabled={loading}
      />

        {/* Rich Text Editor */}
        <div className="prose max-w-none">
      <ReactQuill
        value={content}
        onChange={setContent}
        theme="snow"
        readOnly={loading}
            className="bg-white dark:bg-slate-800 rounded-lg border-blue-300 dark:border-blue-700"
            modules={{
              toolbar: [
                [{ 'header': [1, 2, 3, false] }],
                ['bold', 'italic', 'underline', 'strike'],
                [{ 'list': 'ordered'}, { 'list': 'bullet' }],
                ['link', 'blockquote'],
                [{ 'align': [] }],
                ['clean']
              ],
            }}
          />
        </div>

        {/* Tags Input */}
      <input
          className="w-full px-4 py-3 rounded-lg border border-purple-300 dark:border-purple-700 bg-white/70 dark:bg-slate-800/70 focus:outline-none focus:ring-2 focus:ring-purple-400 dark:focus:ring-purple-600 transition placeholder-gray-400 dark:placeholder-gray-500 text-purple-900 dark:text-purple-100"
        value={tags}
        onChange={e => setTags(e.target.value)}
        placeholder="Tags (comma separated)"
        disabled={loading}
      />

        {/* Type Selection */}
      <select
          className="w-full px-4 py-3 rounded-lg border border-emerald-300 dark:border-emerald-700 bg-white/70 dark:bg-slate-800/70 focus:outline-none focus:ring-2 focus:ring-emerald-400 dark:focus:ring-emerald-600 transition text-emerald-900 dark:text-emerald-100"
          value={type}
          onChange={e => setType(e.target.value)}
        disabled={loading}
      >
          {types.map(t => (
            <option key={t} value={t} className="bg-white dark:bg-slate-800">
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </option>
        ))}
      </select>

        {/* Image Upload Area */}
        <div
          className="border-2 border-dashed border-blue-300 dark:border-blue-700 rounded-lg p-6 text-center transition-colors hover:border-blue-500 dark:hover:border-blue-500"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
      <input
        type="file"
        accept="image/*"
            multiple
            className="hidden"
        onChange={handleImageChange}
        disabled={loading}
            id="image-upload"
          />
          <label
            htmlFor="image-upload"
            className="flex flex-col items-center cursor-pointer"
          >
            <FiImage className="w-8 h-8 text-blue-500 mb-2" />
            <span className="text-gray-600 dark:text-gray-300">
              Drag & drop images here or click to select
            </span>
            <span className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Supports: JPG, PNG, WebP (max 5MB each)
            </span>
          </label>
        </div>

        {/* Image Preview Grid */}
        {images.length > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-4">
            {images.map((img, index) => (
              <div key={index} className="relative group">
                <img
                  src={img.preview}
                  alt={`Preview ${index + 1}`}
                  className="w-full h-32 object-cover rounded-lg"
                />
                <button
                  type="button"
                  onClick={() => removeImage(index)}
                  className="absolute top-2 right-2 bg-red-500 text-white p-1 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <FiX className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Error Display */}
        {(localError || error) && (
          <div className="bg-red-100 dark:bg-red-900/50 text-red-600 dark:text-red-300 p-4 rounded-lg">
            {localError || error}
        </div>
      )}

        {/* Action Buttons */}
        <div className="flex gap-4 justify-end">
          <button
            type="button"
            onClick={() => setShowPreview(!showPreview)}
            className="px-6 py-3 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors flex items-center gap-2"
          >
            <FiEye className="w-5 h-5" />
            {showPreview ? 'Hide Preview' : 'Show Preview'}
        </button>
          
        {onCancel && (
            <button
              type="button"
              className="px-6 py-3 rounded-lg bg-gray-300 text-gray-800 hover:bg-gray-400 transition-colors disabled:opacity-50"
              onClick={onCancel}
              disabled={loading}
            >
            Cancel
          </button>
        )}
          
          <button
            type="submit"
            className="px-6 py-3 rounded-lg bg-gradient-to-r from-blue-500 via-emerald-500 to-purple-500 text-white font-semibold hover:from-blue-600 hover:to-purple-600 transition-colors disabled:opacity-50 flex items-center gap-2"
            disabled={loading}
          >
            {loading ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Saving...
              </>
            ) : (
              'Save Article'
            )}
          </button>
      </div>
    </form>

      {/* Article Preview */}
      {showPreview && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-8 z-50">
          <div className="bg-white dark:bg-slate-900 rounded-2xl shadow-2xl p-8 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold text-blue-600 dark:text-blue-400">{title || 'Untitled Article'}</h2>
              <button
                onClick={() => setShowPreview(false)}
                className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                <FiX className="w-6 h-6" />
              </button>
            </div>
            <div className="prose dark:prose-invert max-w-none">
              <div dangerouslySetInnerHTML={{ __html: content }} />
            </div>
            {tags && (
              <div className="mt-6 flex flex-wrap gap-2">
                {tags.split(',').map((tag, index) => (
                  <span
                    key={index}
                    className="px-3 py-1 rounded-full bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-300 text-sm"
                  >
                    {tag.trim()}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ArticleEditor;