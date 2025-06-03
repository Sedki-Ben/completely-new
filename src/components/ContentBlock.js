import React from 'react';
import { FiX, FiArrowUp, FiArrowDown, FiImage, FiAlignLeft, FiAlignCenter, FiAlignRight, FiAlignJustify } from 'react-icons/fi';
import RichTextEditor from './RichTextEditor';

const BLOCK_TYPES = {
  paragraph: 'Paragraph',
  heading: 'Heading',
  quote: 'Quote',
  image: 'Image',
  'image-group': 'Image Group',
  list: 'List'
};

const HEADING_LEVELS = {
  h2: 'Heading 2',
  h3: 'Heading 3',
  h4: 'Heading 4'
};

const LIST_TYPES = {
  bullet: 'Bullet List',
  numbered: 'Numbered List'
};

const ALIGNMENTS = [
  { value: 'left', icon: FiAlignLeft },
  { value: 'center', icon: FiAlignCenter },
  { value: 'right', icon: FiAlignRight },
  { value: 'justify', icon: FiAlignJustify }
];

const ContentBlock = ({ 
  block, 
  index, 
  onUpdate, 
  onDelete, 
  onMoveUp, 
  onMoveDown,
  isFirst,
  isLast,
  dir = 'ltr'
}) => {
  const handleImageChange = (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      const oversizedFiles = files.filter(file => file.size > 5 * 1024 * 1024);
      if (oversizedFiles.length > 0) {
        alert('Some images are larger than 5MB. Please optimize them first.');
        return;
      }

      const readers = files.map(file => {
        return new Promise((resolve) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result);
          reader.readAsDataURL(file);
        });
      });

      Promise.all(readers).then(results => {
        if (block.type === 'image') {
          onUpdate(index, { 
            ...block, 
            content: results[0],
            metadata: {
              ...block.metadata,
              images: [{
                url: results[0],
                caption: '',
                alignment: block.metadata?.alignment || 'center'
              }]
            }
          });
        } else if (block.type === 'image-group') {
          onUpdate(index, {
            ...block,
            content: 'Image Group',
            metadata: {
              ...block.metadata,
              images: results.map(url => ({
                url,
                caption: '',
                alignment: block.metadata?.alignment || 'center'
              }))
            }
          });
        }
      });
    }
  };

  const updateMetadata = (updates) => {
    onUpdate(index, {
      ...block,
      metadata: {
        ...block.metadata,
        ...updates
      }
    });
  };

  return (
    <div className="relative group bg-white dark:bg-slate-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <select
            value={block.type}
            onChange={(e) => onUpdate(index, { ...block, type: e.target.value })}
            className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-gray-100"
          >
            {Object.entries(BLOCK_TYPES).map(([value, label]) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>

          {block.type === 'heading' && (
            <select
              value={block.metadata?.level || 'h2'}
              onChange={(e) => updateMetadata({ level: e.target.value })}
              className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-gray-100"
            >
              {Object.entries(HEADING_LEVELS).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          )}

          {block.type === 'list' && (
            <select
              value={block.metadata?.listType || 'bullet'}
              onChange={(e) => updateMetadata({ listType: e.target.value })}
              className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-gray-100"
            >
              {Object.entries(LIST_TYPES).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          )}

          <div className="flex items-center gap-1">
            {ALIGNMENTS.map(({ value, icon: Icon }) => (
              <button
                key={value}
                onClick={() => updateMetadata({ alignment: value })}
                className={`p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 ${
                  block.metadata?.alignment === value ? 'text-blue-500' : 'text-gray-500'
                }`}
              >
                <Icon className="w-4 h-4" />
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {!isFirst && (
            <button
              onClick={() => onMoveUp(index)}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              <FiArrowUp />
            </button>
          )}
          {!isLast && (
            <button
              onClick={() => onMoveDown(index)}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              <FiArrowDown />
            </button>
          )}
          <button
            onClick={() => onDelete(index)}
            className="p-2 text-red-500 hover:text-red-600 dark:text-red-400 dark:hover:text-red-300"
          >
            <FiX />
          </button>
        </div>
      </div>

      {(block.type === 'image' || block.type === 'image-group') ? (
        <div className="space-y-4">
          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            className="hidden"
            id={`block-image-${index}`}
            multiple={block.type === 'image-group'}
          />
          <label
            htmlFor={`block-image-${index}`}
            className="block w-full p-4 border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-lg cursor-pointer hover:border-blue-500 dark:hover:border-blue-500"
          >
            {block.metadata?.images?.length > 0 ? (
              <div className={`grid ${block.type === 'image-group' ? 'grid-cols-2 gap-4' : ''}`}>
                {block.metadata.images.map((image, imgIndex) => (
                  <div key={imgIndex} className="relative">
                    <img
                      src={image.url}
                      alt={image.caption || `Image ${imgIndex + 1}`}
                      className="max-h-96 mx-auto rounded-lg"
                    />
                    <input
                      type="text"
                      placeholder="Image caption"
                      value={image.caption || ''}
                      onChange={(e) => {
                        const newImages = [...block.metadata.images];
                        newImages[imgIndex] = { ...image, caption: e.target.value };
                        updateMetadata({ images: newImages });
                      }}
                      className="mt-2 w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-gray-100"
                      dir={dir}
                    />
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center">
                <FiImage className="w-8 h-8 mx-auto text-gray-400" />
                <span className="mt-2 block text-sm text-gray-600 dark:text-gray-400">
                  Click to upload {block.type === 'image-group' ? 'images' : 'an image'}
                </span>
              </div>
            )}
          </label>
        </div>
      ) : (
        <div className="space-y-4">
          <RichTextEditor
            content={block.content}
            onChange={(content) => onUpdate(index, { ...block, content })}
            dir={dir}
            placeholder={
              block.type === 'heading' ? 'Enter heading...' :
              block.type === 'quote' ? 'Enter quote...' :
              block.type === 'list' ? 'Enter list items...' :
              'Enter paragraph text...'
            }
            type={block.type}
            metadata={block.metadata}
          />
          {block.type === 'quote' && (
            <input
              type="text"
              placeholder="Quote source"
              value={block.metadata?.source || ''}
              onChange={(e) => updateMetadata({ source: e.target.value })}
              className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-slate-800 text-gray-900 dark:text-gray-100"
              dir={dir}
            />
          )}
        </div>
      )}

      <div className="mt-4">
        <div className="flex items-center gap-4">
          <input
            type="color"
            value={block.metadata?.style?.textColor || '#000000'}
            onChange={(e) => updateMetadata({ 
              style: { 
                ...block.metadata?.style,
                textColor: e.target.value 
              } 
            })}
            className="w-8 h-8 rounded cursor-pointer"
          />
          <input
            type="color"
            value={block.metadata?.style?.backgroundColor || '#ffffff'}
            onChange={(e) => updateMetadata({ 
              style: { 
                ...block.metadata?.style,
                backgroundColor: e.target.value 
              } 
            })}
            className="w-8 h-8 rounded cursor-pointer"
          />
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-600 dark:text-gray-400">Margin Top:</label>
            <input
              type="number"
              value={block.metadata?.style?.margins?.top || 0}
              onChange={(e) => updateMetadata({
                style: {
                  ...block.metadata?.style,
                  margins: {
                    ...block.metadata?.style?.margins,
                    top: parseInt(e.target.value)
                  }
                }
              })}
              className="w-16 px-2 py-1 rounded border border-gray-300 dark:border-gray-700"
            />
          </div>
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-600 dark:text-gray-400">Margin Bottom:</label>
            <input
              type="number"
              value={block.metadata?.style?.margins?.bottom || 0}
              onChange={(e) => updateMetadata({
                style: {
                  ...block.metadata?.style,
                  margins: {
                    ...block.metadata?.style?.margins,
                    bottom: parseInt(e.target.value)
                  }
                }
              })}
              className="w-16 px-2 py-1 rounded border border-gray-300 dark:border-gray-700"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ContentBlock; 