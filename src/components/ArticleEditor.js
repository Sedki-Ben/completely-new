import React, { useState, useRef, useEffect } from 'react';
import { FiImage, FiX, FiEye, FiBold, FiItalic, FiUnderline, FiGlobe } from 'react-icons/fi';
import { BsTypeH2, BsBlockquoteLeft, BsParagraph, BsListUl, BsListOl } from 'react-icons/bs';

const types = [
  'etoile-du-sahel',
  'the-beautiful-game',
  'all-sports-hub'
];

const languages = [
  { code: 'en', name: 'English', dir: 'ltr' },
  { code: 'fr', name: 'Français', dir: 'ltr' },
  { code: 'ar', name: 'العربية', dir: 'rtl' }
];

// Rich Text Toolbar Component
// Replace RichTextToolbar component (lines 12-78)
// Replace RichTextToolbar component (lines 12-78)
// Add this component before RichTextToolbar (around line 12)
const ColorPicker = ({ onColorSelect, textRef }) => {
  const [showColors, setShowColors] = useState(false);
  const [currentColor, setCurrentColor] = useState('#000000');
  
  const standardColors = [
    { name: 'Black', value: '#000000' },
    { name: 'Dark Gray', value: '#404040' },
    { name: 'Gray', value: '#808080' },
    { name: 'Light Gray', value: '#C0C0C0' },
    { name: 'Red', value: '#FF0000' },
    { name: 'Orange', value: '#FF8000' },
    { name: 'Yellow', value: '#FFFF00' },
    { name: 'Green', value: '#00FF00' },
    { name: 'Blue', value: '#0000FF' },
    { name: 'Purple', value: '#8000FF' },
    { name: 'Dark Red', value: '#800000' },
    { name: 'Dark Blue', value: '#000080' },
    { name: 'Dark Green', value: '#008000' },
    { name: 'Brown', value: '#8B4513' },
    { name: 'Pink', value: '#FF69B4' },
    { name: 'Teal', value: '#008080' }
  ];

  const applyColor = (color) => {
    setCurrentColor(color);
    if (textRef.current) {
      textRef.current.focus();
      document.execCommand('foreColor', false, color);
      setTimeout(() => {
        if (textRef.current) {
          onColorSelect(textRef.current.innerHTML);
        }
      }, 50);
    }
    setShowColors(false);
  };

  return (
    <div className="relative">
      <button
        type="button"
        onMouseDown={(e) => e.preventDefault()}
        onClick={() => setShowColors(!showColors)}
        className="flex items-center gap-1 p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300"
        title="Text Color"
      >
        <div className="w-4 h-4 border border-gray-400 rounded" style={{ backgroundColor: currentColor }}></div>
        <span className="text-xs">▼</span>
      </button>
      
      {showColors && (
        <div className="absolute top-full left-0 z-50 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg shadow-lg p-2 grid grid-cols-4 gap-1 w-48">
          {standardColors.map((color) => (
            <button
              key={color.value}
              type="button"
              onClick={() => applyColor(color.value)}
              className="w-8 h-8 rounded border border-gray-300 hover:border-gray-500 transition-colors"
              style={{ backgroundColor: color.value }}
              title={color.name}
            />
          ))}
          <button
            type="button"
            onClick={() => setShowColors(false)}
            className="col-span-4 mt-2 px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
          >
            Close
          </button>
        </div>
      )}
    </div>
  );
};

const RichTextToolbar = ({ onFormat, textRef }) => {
  const formatText = (command, value = null) => {
    if (!textRef.current) return;
    
    // Ensure the editor is focused
    textRef.current.focus();
    
    try {
      // Execute the command
      const success = document.execCommand(command, false, value);
      
      if (success) {
        // Trigger update after a short delay to ensure DOM is updated
        setTimeout(() => {
          if (textRef.current) {
            onFormat(textRef.current.innerHTML);
          }
        }, 50);
      }
    } catch (error) {
      console.warn('execCommand failed for:', command, error);
    }
  };

  return (
    <div className="flex flex-wrap gap-1 p-2 bg-gray-100 dark:bg-gray-700 rounded-t-lg border-b border-gray-200 dark:border-gray-600">
      <button
        type="button"
        onMouseDown={(e) => e.preventDefault()} // Prevent losing focus
        onClick={() => formatText('bold')}
        className="p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300"
        title="Bold"
      >
        <FiBold className="w-4 h-4" />
      </button>
      <button
        type="button"
        onMouseDown={(e) => e.preventDefault()}
        onClick={() => formatText('italic')}
        className="p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300"
        title="Italic"
      >
        <FiItalic className="w-4 h-4" />
      </button>
      <button
        type="button"
        onMouseDown={(e) => e.preventDefault()}
        onClick={() => formatText('underline')}
        className="p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300"
        title="Underline"
      >
        <FiUnderline className="w-4 h-4" />
      </button>
      <div className="w-px h-6 bg-gray-300 dark:bg-gray-600 mx-1"></div>
      <button
        type="button"
        onMouseDown={(e) => e.preventDefault()}
        onClick={() => formatText('insertUnorderedList')}
        className="p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300"
        title="Bullet List"
      >
        <BsListUl className="w-4 h-4" />
      </button>
      <button
        type="button"
        onMouseDown={(e) => e.preventDefault()}
        onClick={() => formatText('insertOrderedList')}
        className="p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300"
        title="Numbered List"
      >
        <BsListOl className="w-4 h-4" />
      </button>
      <div className="w-px h-6 bg-gray-300 dark:border-gray-600 mx-1"></div>
      <select
        onChange={(e) => {
          formatText('fontSize', e.target.value);
          e.target.blur(); // Remove focus after selection
        }}
        className="px-2 py-1 text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300"
      >
        <option value="3">Normal</option>
        <option value="1">Small</option>
        <option value="4">Large</option>
        <option value="6">X-Large</option>
      </select>
      <ColorPicker onColorSelect={(color) => formatText('foreColor', color)} textRef={textRef} />

    </div>
  );
};

// Rich Text Editor Component
// Replace RichTextEditor component (lines 79-110)
const RichTextEditor = ({ content, onChange, placeholder, className = "", dir }) => {
  const editorRef = useRef(null);

  const handleInput = () => {
    if (editorRef.current) {
      onChange(editorRef.current.innerHTML);
    }
  };

  const handleFormat = (newContent) => {
    onChange(newContent);
  };

  const handleKeyDown = (e) => {
    // Prevent form submission on Enter
    if (e.key === 'Enter') {
      e.stopPropagation(); // Stop the event from bubbling up to the form
  
      // Check if we're in a list context to allow normal list behavior
      const currentElement = e.target.closest('li');
      if (currentElement) {
          // Let the browser handle list item creation
          return;
        }
        // For regular paragraphs, allow normal line break behavior
    }
  };
  

  const handleFocus = () => {
    // Ensure the editor is focused for execCommand to work
    if (editorRef.current) {
      editorRef.current.focus();
    }
  };

  useEffect(() => {
    if (editorRef.current && editorRef.current.innerHTML !== content) {
      editorRef.current.innerHTML = content || '';
    }
  }, [content]);

  return (
    <div className={`border border-gray-300 dark:border-gray-600 rounded-lg ${className}`}>
      <RichTextToolbar onFormat={handleFormat} textRef={editorRef} />
      <div
        ref={editorRef}
        contentEditable
        onInput={handleInput}
        onKeyDown={handleKeyDown}
        onFocus={handleFocus}
        className="w-full px-4 py-3 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-b-lg focus:outline-none min-h-[100px]"
        style={{ minHeight: '100px' }}
        data-placeholder={placeholder}
        suppressContentEditableWarning={true}
        dir={dir}
      />
    </div>
  );
};

const ContentBlock = ({ block, onUpdate, onDelete, index, dir }) => {
  const [draggedImageIndex, setDraggedImageIndex] = useState(null);

  const handleContentChange = (content) => {
    onUpdate(index, { ...block, content });
  };

  const handleTextContentChange = (e) => {
    onUpdate(index, { ...block, content: e.target.value });
  };

  const handleMetadataChange = (key, value) => {
    onUpdate(index, {
      ...block,
      metadata: { ...block.metadata, [key]: value }
    });
  };

  const handleImageUpload = (e) => {
    const files = Array.from(e.target.files).slice(0, 3);
    const newImages = files.map(file => ({
      file,
      url: URL.createObjectURL(file),
      caption: '',
      alignment: 'center',
      size: 'medium'
    }));
    
    const existingImages = block.metadata.images || [];
    const combinedImages = [...existingImages, ...newImages].slice(0, 3);
    
    handleMetadataChange('images', combinedImages);
  };

  const updateImage = (imgIndex, updates) => {
    const newImages = [...(block.metadata.images || [])];
    newImages[imgIndex] = { ...newImages[imgIndex], ...updates };
    handleMetadataChange('images', newImages);
  };

  const removeImage = (imgIndex) => {
    const newImages = (block.metadata.images || []).filter((_, i) => i !== imgIndex);
    handleMetadataChange('images', newImages);
  };

  const moveImage = (fromIndex, toIndex) => {
    const newImages = [...(block.metadata.images || [])];
    const [movedImage] = newImages.splice(fromIndex, 1);
    newImages.splice(toIndex, 0, movedImage);
    handleMetadataChange('images', newImages);
  };

  const getImageLayoutClass = (imageCount, imgIndex) => {
    if (imageCount === 1) return 'w-full';
    if (imageCount === 2) return 'w-1/2';
    if (imageCount === 3) return 'w-1/3';
    return 'w-full';
  };

  const renderBlockContent = () => {
    switch (block.type) {
      case 'heading':
        return (
          <div className="space-y-2">
            <textarea
              className="w-full px-4 py-3 text-xl font-bold bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-lg border border-blue-300 dark:border-blue-700 focus:ring-2 focus:ring-blue-400"
              value={block.content}
              onChange={handleTextContentChange}
              rows={1}
              style={{ resize: 'none' }}
              dir={dir}
            />
            <select
              value={block.metadata.level || 2}
              onChange={(e) => handleMetadataChange('level', parseInt(e.target.value))}
              className="px-2 py-1 rounded border border-blue-300 dark:border-blue-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
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
              className="w-full px-4 py-3 italic bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-lg border border-blue-300 dark:border-blue-700 focus:ring-2 focus:ring-blue-400"
              value={block.content}
              onChange={handleTextContentChange}
              rows={3}
              dir={dir}
            />
            <input
              type="text"
              placeholder="Quote source"
              value={block.metadata.source || ''}
              onChange={(e) => handleMetadataChange('source', e.target.value)}
              className="w-full px-4 py-2 rounded-lg border border-blue-300 dark:border-blue-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
              dir={dir}
            />
          </div>
        );

      case 'image-group':
        const images = block.metadata.images || [];
        return (
          <div className="space-y-4">
            {/* Image Upload Area */}
            <div className="border-2 border-dashed border-blue-300 dark:border-blue-700 rounded-lg p-4">
              <input
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                onChange={handleImageUpload}
                id={`image-upload-${index}`}
              />
              <label
                htmlFor={`image-upload-${index}`}
                className="flex flex-col items-center cursor-pointer text-center"
              >
                <FiImage className="w-8 h-8 text-blue-500 mb-2" />
                <span className="text-gray-600 dark:text-gray-300">
                  Upload images (max 3) - {images.length}/3 uploaded
                </span>
              </label>
            </div>

            {/* Image Preview and Controls */}
            {images.length > 0 && (
              <div className="flex gap-4 flex-wrap">
                {images.map((img, imgIndex) => (
                  <div
                    key={imgIndex}
                    className={`relative group ${getImageLayoutClass(images.length, imgIndex)}`}
                    draggable
                    onDragStart={() => setDraggedImageIndex(imgIndex)}
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={(e) => {
                      e.preventDefault();
                      if (draggedImageIndex !== null && draggedImageIndex !== imgIndex) {
                        moveImage(draggedImageIndex, imgIndex);
                        setDraggedImageIndex(null);
                      }
                    }}
                  >
                    <img
                      src={img.url}
                      alt=""
                      className={`w-full object-cover rounded-lg cursor-move ${
                        img.size === 'small' ? 'h-32' : 
                        img.size === 'large' ? 'h-64' : 'h-48'
                      }`}
                    />
                    
                    {/* Image Controls */}
                    <div className="absolute top-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        type="button"
                        onClick={() => removeImage(imgIndex)}
                        className="bg-red-500 text-white p-1 rounded-full"
                      >
                        <FiX className="w-3 h-3" />
                      </button>
                    </div>

                    {/* Size Controls */}
                    <div className="absolute bottom-2 left-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        type="button"
                        onClick={() => updateImage(imgIndex, { size: 'small' })}
                        className={`px-2 py-1 text-xs rounded ${img.size === 'small' ? 'bg-blue-500 text-white' : 'bg-white/80 text-blue-500'}`}
                      >
                        S
                      </button>
                      <button
                        type="button"
                        onClick={() => updateImage(imgIndex, { size: 'medium' })}
                        className={`px-2 py-1 text-xs rounded ${img.size === 'medium' ? 'bg-blue-500 text-white' : 'bg-white/80 text-blue-500'}`}
                      >
                        M
                      </button>
                      <button
                        type="button"
                        onClick={() => updateImage(imgIndex, { size: 'large' })}
                        className={`px-2 py-1 text-xs rounded ${img.size === 'large' ? 'bg-blue-500 text-white' : 'bg-white/80 text-blue-500'}`}
                      >
                        L
                      </button>
                    </div>

                    {/* Caption Input */}
                    <input
                      type="text"
                      placeholder="Caption"
                      value={img.caption || ''}
                      onChange={(e) => updateImage(imgIndex, { caption: e.target.value })}
                      className="mt-2 w-full px-3 py-1 text-sm rounded border border-blue-300 dark:border-blue-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
                      dir={dir}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        );

      default: // paragraph with rich text
        return (
          <RichTextEditor
            content={block.content}
            onChange={handleContentChange}
            placeholder="Write your paragraph content here..."
            className="w-full"
            dir={dir}
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

// Preview Component
const ArticlePreview = ({ title, mainImage, blocks, tags, type, language }) => {
  const renderPreviewBlock = (block, index) => {
    switch (block.type) {
      case 'heading':
        const HeadingTag = `h${block.metadata.level || 2}`;
        return (
          <HeadingTag key={index} className={`font-bold text-gray-900 dark:text-gray-100 mb-4 ${
            block.metadata.level === 2 ? 'text-2xl' :
            block.metadata.level === 3 ? 'text-xl' : 'text-lg'
          }`}>
            {block.content}
          </HeadingTag>
        );
      
      case 'quote':
        return (
          <blockquote key={index} className="border-l-4 border-blue-500 pl-4 italic text-gray-700 dark:text-gray-300 mb-4">
            <div dangerouslySetInnerHTML={{ __html: block.content }} />
            {block.metadata.source && (
              <cite className="block text-sm text-gray-500 dark:text-gray-400 mt-2">
                — {block.metadata.source}
              </cite>
            )}
          </blockquote>
        );
      
      case 'image-group':
        const images = block.metadata.images || [];
        if (images.length === 0) return null;
        
        return (
          <div key={index} className="mb-6">
            <div className={`flex gap-4 ${images.length === 1 ? 'justify-center' : 'justify-between'}`}>
              {images.map((img, imgIndex) => (
                <div key={imgIndex} className={`${
                  images.length === 1 ? 'w-full max-w-2xl' :
                  images.length === 2 ? 'w-1/2' : 'w-1/3'
                }`}>
                  <img
                    src={img.url}
                    alt=""
                    className={`w-full object-cover rounded-lg ${
                      img.size === 'small' ? 'h-32' :
                      img.size === 'large' ? 'h-64' : 'h-48'
                    }`}
                  />
                  {img.caption && (
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-2 text-center">
                      {img.caption}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        );
      
      default: // paragraph
        return (
          <div 
            key={index} 
            className="mb-4 text-gray-900 dark:text-gray-100"
            dangerouslySetInnerHTML={{ __html: block.content }}
          />
        );
    }
  };

  return (
    <div className="bg-white dark:bg-gray-900 p-6 rounded-lg border border-gray-200 dark:border-gray-700">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-4">
        {title || 'Article Preview'}
      </h1>
      
      {mainImage && (
        <img
          src={mainImage.preview || mainImage}
          alt="Main article"
          className="w-full h-64 object-cover rounded-lg mb-6"
        />
      )}
      
      <div className="space-y-4">
        {blocks.map((block, index) => renderPreviewBlock(block, index))}
      </div>
      
      {tags && (
        <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="flex flex-wrap gap-2">
            {tags.split(',').map((tag, index) => {
              // Theme colors based on article type
              const themeColors = {
                'etoile-du-sahel': 'px-3 py-1 bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 rounded-full text-sm font-medium transition-colors cursor-pointer hover:text-red-900 dark:hover:text-red-400',
                'the-beautiful-game': 'px-3 py-1 bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 rounded-full text-sm font-medium transition-colors cursor-pointer hover:text-green-900 dark:hover:text-green-400',
                'all-sports-hub': 'px-3 py-1 bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 rounded-full text-sm font-medium transition-colors cursor-pointer hover:text-purple-900 dark:hover:text-purple-400',
                'archive': 'px-3 py-1 bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200 rounded-full text-sm font-medium transition-colors cursor-pointer hover:text-yellow-900 dark:hover:text-yellow-400',
                default: 'px-3 py-1 bg-gray-100 dark:bg-gray-900 text-gray-800 dark:text-gray-200 rounded-full text-sm font-medium transition-colors cursor-pointer hover:text-gray-900 dark:hover:text-gray-400'
              };
              const tagClass = themeColors[type] || themeColors.default;
              
              return (
                <span
                  key={index}
                  className={tagClass}
                >
                  #{tag.trim()}
                </span>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

const ArticleEditor = ({ onSave, onCancel, initialData = {}, loading = false, error = '', userRole }) => {
  // Multilingual content state
  const [currentLanguage, setCurrentLanguage] = useState('en');
  const [titles, setTitles] = useState({
    en: initialData?.titles?.en || '',
    fr: initialData?.titles?.fr || '',
    ar: initialData?.titles?.ar || ''
  });
  const [contentBlocks, setContentBlocks] = useState({
    en: initialData?.content?.en || [],
    fr: initialData?.content?.fr || [],
    ar: initialData?.content?.ar || []
  });
  
  const [mainImage, setMainImage] = useState(initialData?.image || null);
  const [tags, setTags] = useState(initialData?.tags ? initialData.tags.join(', ') : '');
  const [type, setType] = useState(initialData?.type || types[0]);
  const [localError, setLocalError] = useState('');
  const [showPreview, setShowPreview] = useState(false);
  const [draggedBlockIndex, setDraggedBlockIndex] = useState(null);

  // Permission check
  const hasPermission = ['writer', 'admin'].includes(userRole);

  const getCurrentLanguageInfo = () => languages.find(lang => lang.code === currentLanguage);

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

  const handleTitleChange = (value) => {
    setTitles(prev => ({
      ...prev,
      [currentLanguage]: value
    }));
  };

  const handleBlockUpdate = (index, updatedBlock) => {
    setContentBlocks(prev => ({
      ...prev,
      [currentLanguage]: prev[currentLanguage].map((block, i) => 
        i === index ? updatedBlock : block
      )
    }));
  };

  const handleBlockDelete = (index) => {
    setContentBlocks(prev => ({
      ...prev,
      [currentLanguage]: prev[currentLanguage].filter((_, i) => i !== index)
    }));
  };

  const moveBlock = (fromIndex, toIndex) => {
    setContentBlocks(prev => {
      const newBlocks = [...prev[currentLanguage]];
  const [movedBlock] = newBlocks.splice(fromIndex, 1);
  newBlocks.splice(toIndex, 0, movedBlock);
      return {
        ...prev,
        [currentLanguage]: newBlocks
      };
    });
};

  const addBlock = (blockType) => {
    const newBlock = {
      type: blockType,
      content: '',
      metadata: blockType === 'heading' ? { level: 2 } :
                blockType === 'quote' ? { source: '' } :
                blockType === 'image-group' ? { images: [] } :
                {}
    };
    setContentBlocks(prev => ({
      ...prev,
      [currentLanguage]: [...prev[currentLanguage], newBlock]
    }));
  };

  // Convert rich text content to plain text for excerpt
  const stripHtml = (html) => {
    const tmp = document.createElement('div');
    tmp.innerHTML = html;
    return tmp.textContent || tmp.innerText || '';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate that all languages have titles
    const missingTitles = languages.filter(lang => !titles[lang.code]?.trim());
    if (missingTitles.length > 0) {
      setLocalError(`Title is required for all languages. Missing: ${missingTitles.map(l => l.name).join(', ')}`);
      return;
    }

    if (!mainImage) {
      setLocalError('Main image is required.');
      return;
    }

    // Validate that each language has at least one paragraph
    const missingContent = languages.filter(lang => {
      const blocks = contentBlocks[lang.code] || [];
      const firstParagraph = blocks.find(b => b.type === 'paragraph' && stripHtml(b.content).trim());
      return !firstParagraph;
    });

    if (missingContent.length > 0) {
      setLocalError(`At least one paragraph is required for all languages. Missing content: ${missingContent.map(l => l.name).join(', ')}`);
      return;
    }
    
    setLocalError('');
    
    const formData = new FormData();
    
    // Collect all image files from all languages
    const contentImageFiles = [];
    languages.forEach(lang => {
      const blocks = contentBlocks[lang.code] || [];
      blocks.forEach(block => {
        if (block.type === 'image-group' && block.metadata?.images) {
          block.metadata.images.forEach(img => {
            if (img.file) {
              contentImageFiles.push(img.file);
            }
          });
        }
      });
    });

    // Create translations object with all languages
    const translations = {};
    languages.forEach(lang => {
      const blocks = contentBlocks[lang.code] || [];
      const firstParagraph = blocks.find(b => b.type === 'paragraph' && stripHtml(b.content).trim())?.content || '';
      
      translations[lang.code] = {
        title: titles[lang.code]?.trim() || '',
        excerpt: stripHtml(firstParagraph).slice(0, 200),
        content: blocks.map(b => {
          if (b.type === 'image-group') {
            return {
              ...b,
              content: b.content || 'Image Group'
            };
          }
          return {
            ...b,
            content: b.content || ''
          };
        }).filter(b => {
          if (b.type === 'image-group') {
            return b.metadata?.images?.length > 0;
          }
          return stripHtml(b.content).trim().length > 0;
        })
      };
    });

    formData.append('translations', JSON.stringify(translations));
    
    // Add main image
    if (mainImage?.file) {
      formData.append('image', mainImage.file);
    } else if (typeof mainImage === 'string') {
      formData.append('existingImage', mainImage);
    }
    
    // Add content images
    contentImageFiles.forEach((file, index) => {
      formData.append('contentImages', file);
    });
    
    formData.append('category', type);
    formData.append('status', 'published');
    formData.append('authorImage', '/images/default-author.jpg');
    
    const tagsArray = tags.split(',').map(t => t.trim()).filter(Boolean);
    if (tagsArray.length > 0) {
      formData.append('tags', JSON.stringify(tagsArray));
    }

    try {
      await onSave(formData);
    } catch (error) {
      if (error.response?.data?.message) {
        setLocalError(error.response.data.message);
        if (error.response.data.field === 'title') {
          const titleInput = document.querySelector('input[placeholder*="Title"]');
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

  const currentLangInfo = getCurrentLanguageInfo();
  const currentBlocks = contentBlocks[currentLanguage] || [];

  return (
    <div className={`flex gap-6 ${showPreview ? 'max-w-7xl' : 'max-w-4xl'} mx-auto`}>
      {/* Editor Panel */}
      <div className={`${showPreview ? 'w-1/2' : 'w-full'} bg-white/80 dark:bg-slate-900/80 backdrop-blur-lg rounded-2xl shadow-2xl p-8 border border-blue-200 dark:border-blue-900`}>
      <form onSubmit={handleSubmit} onKeyDown={(e) => {
        if (e.key === 'Enter' && e.target.tagName !== 'BUTTON' && !e.target.closest('[contenteditable]')) {
          e.preventDefault();
        }
      }} className="space-y-6">
          
          {/* Language Tabs */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-2">
              <FiGlobe className="w-5 h-5 text-gray-600 dark:text-gray-400" />
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Language:</span>
            </div>
            <div className="flex bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
              {languages.map(lang => (
                <button
                  key={lang.code}
                  type="button"
                  onClick={() => setCurrentLanguage(lang.code)}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                    currentLanguage === lang.code
                      ? 'bg-blue-500 text-white shadow-sm'
                      : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                  }`}
                >
                  {lang.name}
                </button>
              ))}
            </div>
          </div>

          {/* Title Input */}
          <input
            className={`w-full px-4 py-3 rounded-lg border border-blue-300 dark:border-blue-700 bg-white/70 dark:bg-slate-800/70 focus:outline-none focus:ring-2 focus:ring-blue-400 dark:focus:ring-blue-600 transition placeholder-gray-400 dark:placeholder-gray-500 text-blue-900 dark:text-blue-100`}
            value={titles[currentLanguage]}
            onChange={(e) => handleTitleChange(e.target.value)}
            placeholder={`Article Title (${currentLangInfo.name})`}
            required
            disabled={loading}
            dir={currentLangInfo.dir}
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
                    src={typeof mainImage === 'string' ? mainImage : mainImage.preview}
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
          <div className="space-y-6" dir={currentLangInfo.dir}>
            {currentBlocks.map((block, index) => (
            <div
              key={index}
              className="relative group flex items-start gap-2"
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                if (draggedBlockIndex !== null && draggedBlockIndex !== index) {
                  moveBlock(draggedBlockIndex, index);
                  setDraggedBlockIndex(null);
                }
              }}
            >
              {/* Drag Handle */}
              <div
                  className={`flex-shrink-0 mt-4 opacity-0 group-hover:opacity-100 transition-opacity cursor-move ${currentLangInfo.dir === 'rtl' ? 'order-2' : ''}`}
                draggable
                onDragStart={() => setDraggedBlockIndex(index)}
                onDragEnd={() => setDraggedBlockIndex(null)}
              >
                <div className="w-6 h-6 bg-gray-300 dark:bg-gray-600 rounded flex items-center justify-center hover:bg-gray-400 dark:hover:bg-gray-500">
                  <svg className="w-4 h-4 text-gray-600 dark:text-gray-300" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M10 6a2 2 0 110-4 2 2 0 010 4zM10 12a2 2 0 110-4 2 2 0 010 4zM10 18a2 2 0 110-4 2 2 0 010 4z"></path>
                  </svg>
                </div>
              </div>

              {/* Content Block */}
              <div className="flex-1">
                <ContentBlock
                  block={block}
                  index={index}
                  onUpdate={handleBlockUpdate}
                  onDelete={handleBlockDelete}
                    dir={currentLangInfo.dir}
                />
              </div>
            </div>
          ))}
          </div>

          {/* Add Block Buttons */}
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
            onChange={(e) => setTags(e.target.value)}
            placeholder="Tags (comma separated)"
            disabled={loading}
          />

          {/* Type Selection */}
          <select
            className="w-full px-4 py-3 rounded-lg border border-emerald-300 dark:border-emerald-700 bg-white/70 dark:bg-slate-800/70 focus:outline-none focus:ring-2 focus:ring-emerald-400 dark:focus:ring-emerald-600 transition text-emerald-900 dark:text-emerald-100"
            value={type}
            onChange={(e) => setType(e.target.value)}
            disabled={loading}
          >
            {types.map(t => (
              <option key={t} value={t} className="bg-white dark:bg-slate-800">
                {t.charAt(0).toUpperCase() + t.slice(1).replace(/-/g, ' ')}
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

      {/* Preview Panel */}
      {showPreview && (
        <div className="w-1/2 bg-white/80 dark:bg-slate-900/80 backdrop-blur-lg rounded-2xl shadow-2xl p-8 border border-blue-200 dark:border-blue-900">
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
            <FiEye className="w-5 h-5" />
            Live Preview ({currentLangInfo.name})
          </h2>
          <div className="max-h-[80vh] overflow-y-auto" dir={currentLangInfo.dir}>
            <ArticlePreview
              title={titles[currentLanguage]}
              mainImage={mainImage}
              blocks={currentBlocks}
              tags={tags}
              type={type}
              language={currentLanguage}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default ArticleEditor;