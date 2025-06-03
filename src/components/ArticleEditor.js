import React, { useState, useCallback, useRef } from 'react';
import { FiImage, FiX, FiEye, FiPlus, FiAlignLeft, FiAlignCenter, FiAlignRight } from 'react-icons/fi';
import { BsTypeH2, BsTypeH3, BsBlockquoteLeft, BsParagraph } from 'react-icons/bs';

const types = [
  'etoile-du-sahel',
  'the-beautiful-game',
  'all-sports-hub'
];

const ContentBlock = ({ block, onUpdate, onDelete, index }) => {
  const [isEditing, setIsEditing] = useState(false);

  const handleContentChange = (e) => {
    onUpdate(index, { ...block, content: e.target.value });
  };

  const handleMetadataChange = (key, value) => {
    onUpdate(index, {
      ...block,
      metadata: { ...block.metadata, [key]: value }
    });
  };

  const renderBlockContent = () => {
    switch (block.type) {
      case 'heading':
        return (
          <div className="space-y-2">
            <textarea
              className="w-full px-4 py-3 text-xl font-bold bg-white/70 dark:bg-slate-800/70 rounded-lg border border-blue-300 dark:border-blue-700 focus:ring-2 focus:ring-blue-400"
              value={block.content}
              onChange={handleContentChange}
              rows={1}
              style={{ resize: 'none' }}
            />
            <select
              value={block.metadata.level || 2}
              onChange={(e) => handleMetadataChange('level', parseInt(e.target.value))}
              className="px-2 py-1 rounded border border-blue-300 dark:border-blue-700 bg-white/70 dark:bg-slate-800/70"
            >
              <option value={2}>H2</option>
              <option value={3}>H3</option>
              <option value={4}>H4</option>
            </select>
          </div>
        );

      case 'quote':
        return (
          <div className="space-y-2">
            <textarea
              className="w-full px-4 py-3 italic bg-white/70 dark:bg-slate-800/70 rounded-lg border border-blue-300 dark:border-blue-700 focus:ring-2 focus:ring-blue-400"
              value={block.content}
              onChange={handleContentChange}
              rows={3}
            />
            <input
              type="text"
              placeholder="Quote source"
              value={block.metadata.source || ''}
              onChange={(e) => handleMetadataChange('source', e.target.value)}
              className="w-full px-4 py-2 rounded-lg border border-blue-300 dark:border-blue-700 bg-white/70 dark:bg-slate-800/70"
            />
          </div>
        );

      case 'image':
      case 'image-group':
        return (
          <div className="space-y-4">
            <div className="flex gap-4 flex-wrap">
              {block.metadata.images.map((img, imgIndex) => (
                <div key={imgIndex} className="relative group">
                  <img
                    src={img.url}
                    alt=""
                    className="max-w-xs h-32 object-cover rounded-lg"
                  />
                  <div className="absolute bottom-2 right-2 flex gap-2">
                    <button
                      type="button"
                      onClick={() => handleMetadataChange('alignment', 'left')}
                      className={`p-1 rounded ${img.alignment === 'left' ? 'bg-blue-500 text-white' : 'bg-white/80 text-blue-500'}`}
                    >
                      <FiAlignLeft />
                    </button>
                    <button
                      type="button"
                      onClick={() => handleMetadataChange('alignment', 'center')}
                      className={`p-1 rounded ${img.alignment === 'center' ? 'bg-blue-500 text-white' : 'bg-white/80 text-blue-500'}`}
                    >
                      <FiAlignCenter />
                    </button>
                    <button
                      type="button"
                      onClick={() => handleMetadataChange('alignment', 'right')}
                      className={`p-1 rounded ${img.alignment === 'right' ? 'bg-blue-500 text-white' : 'bg-white/80 text-blue-500'}`}
                    >
                      <FiAlignRight />
                    </button>
                  </div>
                  <input
                    type="text"
                    placeholder="Caption"
                    value={img.caption || ''}
                    onChange={(e) => {
                      const newImages = [...block.metadata.images];
                      newImages[imgIndex] = { ...img, caption: e.target.value };
                      handleMetadataChange('images', newImages);
                    }}
                    className="mt-2 w-full px-3 py-1 text-sm rounded border border-blue-300 dark:border-blue-700 bg-white/70 dark:bg-slate-800/70"
                  />
                </div>
              ))}
            </div>
          </div>
        );

      default: // paragraph
        return (
          <textarea
            className="w-full px-4 py-3 bg-white/70 dark:bg-slate-800/70 rounded-lg border border-blue-300 dark:border-blue-700 focus:ring-2 focus:ring-blue-400"
            value={block.content}
            onChange={handleContentChange}
            rows={4}
          />
        );
    }
  };

  return (
    <div className="relative group p-4 rounded-lg border border-transparent hover:border-blue-300 dark:hover:border-blue-700">
      {renderBlockContent()}
      <button
        type="button"
        onClick={() => onDelete(index)}
        className="absolute -right-2 -top-2 bg-red-500 text-white p-1 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
      >
        <FiX className="w-4 h-4" />
      </button>
    </div>
  );
};

const ArticleEditor = ({ onSave, onCancel, initialData = {}, loading = false, error = '', userRole }) => {
  const [title, setTitle] = useState(initialData.title || '');
  const [mainImage, setMainImage] = useState(initialData.image || null);
  const [blocks, setBlocks] = useState(initialData.content || []);
  const [tags, setTags] = useState(initialData.tags ? initialData.tags.join(', ') : '');
  const [type, setType] = useState(initialData.type || types[0]);
  const [localError, setLocalError] = useState('');
  const [showPreview, setShowPreview] = useState(false);
  const [dragging, setDragging] = useState(false);

  // Permission check
  const hasPermission = ['writer', 'admin'].includes(userRole);

  const handleMainImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.size > 5 * 1024 * 1024) {
        setLocalError('Main image size should be less than 5MB.');
        return;
      }
      setMainImage({
        file,
        preview: URL.createObjectURL(file)
      });
    }
  };

  const handleBlockUpdate = (index, updatedBlock) => {
    const newBlocks = [...blocks];
    newBlocks[index] = updatedBlock;
    setBlocks(newBlocks);
  };

  const handleBlockDelete = (index) => {
    setBlocks(blocks.filter((_, i) => i !== index));
  };

  const addBlock = (type) => {
    const newBlock = {
      type,
      content: '',
      metadata: type === 'heading' ? { level: 2 } :
                type === 'quote' ? { source: '' } :
                type === 'image-group' ? { images: [] } :
                {}
    };
    setBlocks([...blocks, newBlock]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!title || !mainImage) {
      setLocalError('Title and main image are required.');
      return;
    }

    // Validate that there's at least one non-empty paragraph for excerpt
    const firstParagraph = blocks.find(b => b.type === 'paragraph' && b.content.trim())?.content || '';
    if (!firstParagraph) {
      setLocalError('At least one paragraph is required for the article excerpt.');
      return;
    }

    // Validate that all content blocks have content
    const emptyBlock = blocks.find(b => !b.content.trim());
    if (emptyBlock) {
      setLocalError('All content blocks must have content.');
      return;
    }
    
    setLocalError('');
    
    const formData = new FormData();
    
    // Add translations with required fields
    const translations = {
      en: {
        title: title.trim(),
        excerpt: firstParagraph.slice(0, 200), // Use first 200 chars of first paragraph as excerpt
        content: blocks.filter(b => b.content.trim()) // Remove any empty blocks
      },
      ar: {
        title: 'قيد الترجمة', // "Under Translation" in Arabic
        excerpt: 'قيد الترجمة', // "Under Translation" in Arabic
        content: [] // Will be translated later
      }
    };

    // Ensure all required fields are properly stringified
    formData.append('translations', JSON.stringify(translations));
    
    // Add the main image with the correct field name
    if (mainImage?.file) {
      formData.append('image', mainImage.file);
    }
    
    // Add other fields
    formData.append('category', type);
    formData.append('status', 'draft');
    formData.append('authorImage', '/images/default-author.jpg');
    
    // Add tags if present
    const tagsArray = tags.split(',').map(t => t.trim()).filter(Boolean);
    if (tagsArray.length > 0) {
      formData.append('tags', JSON.stringify(tagsArray));
    }

    try {
      await onSave(formData);
    } catch (error) {
      if (error.response?.data?.message) {
        setLocalError(error.response.data.message);
        // If it's a title error, highlight the title input
        if (error.response.data.field === 'title') {
          const titleInput = document.querySelector('input[placeholder="Article Title"]');
          if (titleInput) {
            titleInput.classList.add('border-red-500', 'focus:ring-red-500');
            titleInput.focus();
          }
        }
      } else {
        setLocalError('An error occurred while saving the article. Please try again.');
      }
    }
  };

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

        {/* Main Image Upload */}
        <div className="space-y-4">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Main Article Image
          </label>
          <div className="relative">
            {mainImage ? (
              <div className="relative group">
                <img
                  src={mainImage.preview}
                  alt="Main article"
                  className="w-full h-64 object-cover rounded-lg"
                />
                <button
                  type="button"
                  onClick={() => setMainImage(null)}
                  className="absolute top-2 right-2 bg-red-500 text-white p-1 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <FiX className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <div className="border-2 border-dashed border-blue-300 dark:border-blue-700 rounded-lg p-6 text-center">
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleMainImageChange}
                  disabled={loading}
                  id="main-image-upload"
                />
                <label
                  htmlFor="main-image-upload"
                  className="flex flex-col items-center cursor-pointer"
                >
                  <FiImage className="w-8 h-8 text-blue-500 mb-2" />
                  <span className="text-gray-600 dark:text-gray-300">
                    Upload main article image
                  </span>
                </label>
              </div>
            )}
          </div>
        </div>

        {/* Content Blocks */}
        <div className="space-y-6">
          {blocks.map((block, index) => (
            <ContentBlock
              key={index}
              block={block}
              index={index}
              onUpdate={handleBlockUpdate}
              onDelete={handleBlockDelete}
            />
          ))}
        </div>

        {/* Add Block Button */}
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => addBlock('paragraph')}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 hover:bg-blue-200 dark:hover:bg-blue-800"
          >
            <BsParagraph /> Add Paragraph
          </button>
          <button
            type="button"
            onClick={() => addBlock('heading')}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-purple-100 dark:bg-purple-900 text-purple-700 dark:text-purple-300 hover:bg-purple-200 dark:hover:bg-purple-800"
          >
            <BsTypeH2 /> Add Heading
          </button>
          <button
            type="button"
            onClick={() => addBlock('quote')}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-100 dark:bg-emerald-900 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-200 dark:hover:bg-emerald-800"
          >
            <BsBlockquoteLeft /> Add Quote
          </button>
          <button
            type="button"
            onClick={() => addBlock('image-group')}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-amber-100 dark:bg-amber-900 text-amber-700 dark:text-amber-300 hover:bg-amber-200 dark:hover:bg-amber-800"
          >
            <FiImage /> Add Images
          </button>
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
    </div>
  );
};

export default ArticleEditor;