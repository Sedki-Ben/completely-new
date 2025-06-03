import React from 'react';
import { useEditor, EditorContent } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import Image from '@tiptap/extension-image';
import TextAlign from '@tiptap/extension-text-align';
import Link from '@tiptap/extension-link';
import Placeholder from '@tiptap/extension-placeholder';
import TextStyle from '@tiptap/extension-text-style';
import Color from '@tiptap/extension-color';
import { 
  FiBold, FiItalic, FiUnderline, FiList, FiAlignLeft, 
  FiAlignCenter, FiAlignRight, FiAlignJustify, FiLink, 
  FiImage, FiRotateCcw, FiRotateCw, FiType, FiDroplet
} from 'react-icons/fi';

const MenuBar = ({ editor, type, metadata }) => {
  if (!editor) {
    return null;
  }

  const addImage = () => {
    const url = window.prompt('Enter the image URL');
    if (url) {
      editor.chain().focus().setImage({ src: url }).run();
    }
  };

  const setLink = () => {
    const url = window.prompt('Enter the URL');
    if (url) {
      editor.chain().focus().setLink({ href: url }).run();
    }
  };

  const getHeadingLevel = () => {
    if (type === 'heading') {
      return parseInt(metadata?.level?.substring(1)) || 2;
    }
    return null;
  };

  return (
    <div className="flex flex-wrap gap-2 p-2 border-b border-gray-200 dark:border-gray-700 bg-white/90 dark:bg-slate-800/90">
      {/* Text Style Controls */}
      <div className="flex items-center gap-1 border-r border-gray-200 dark:border-gray-700 pr-2">
        <button
          onClick={() => editor.chain().focus().toggleBold().run()}
          className={`p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 ${
            editor.isActive('bold') ? 'bg-gray-200 dark:bg-gray-700' : ''
          }`}
        >
          <FiBold className="w-4 h-4" />
        </button>
        <button
          onClick={() => editor.chain().focus().toggleItalic().run()}
          className={`p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 ${
            editor.isActive('italic') ? 'bg-gray-200 dark:bg-gray-700' : ''
          }`}
        >
          <FiItalic className="w-4 h-4" />
        </button>
        <button
          onClick={() => editor.chain().focus().toggleStrike().run()}
          className={`p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 ${
            editor.isActive('strike') ? 'bg-gray-200 dark:bg-gray-700' : ''
          }`}
        >
          <FiUnderline className="w-4 h-4" />
        </button>
      </div>

      {/* List Controls */}
      {type !== 'heading' && type !== 'quote' && (
        <div className="flex items-center gap-1 border-r border-gray-200 dark:border-gray-700 pr-2">
          <button
            onClick={() => editor.chain().focus().toggleBulletList().run()}
            className={`p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 ${
              editor.isActive('bulletList') ? 'bg-gray-200 dark:bg-gray-700' : ''
            }`}
          >
            <FiList className="w-4 h-4" />
          </button>
          <button
            onClick={() => editor.chain().focus().toggleOrderedList().run()}
            className={`p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 ${
              editor.isActive('orderedList') ? 'bg-gray-200 dark:bg-gray-700' : ''
            }`}
          >
            <FiList className="w-4 h-4 transform rotate-180" />
          </button>
        </div>
      )}

      {/* Alignment Controls */}
      <div className="flex items-center gap-1 border-r border-gray-200 dark:border-gray-700 pr-2">
        <button
          onClick={() => editor.chain().focus().setTextAlign('left').run()}
          className={`p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 ${
            editor.isActive({ textAlign: 'left' }) ? 'bg-gray-200 dark:bg-gray-700' : ''
          }`}
        >
          <FiAlignLeft className="w-4 h-4" />
        </button>
        <button
          onClick={() => editor.chain().focus().setTextAlign('center').run()}
          className={`p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 ${
            editor.isActive({ textAlign: 'center' }) ? 'bg-gray-200 dark:bg-gray-700' : ''
          }`}
        >
          <FiAlignCenter className="w-4 h-4" />
        </button>
        <button
          onClick={() => editor.chain().focus().setTextAlign('right').run()}
          className={`p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 ${
            editor.isActive({ textAlign: 'right' }) ? 'bg-gray-200 dark:bg-gray-700' : ''
          }`}
        >
          <FiAlignRight className="w-4 h-4" />
        </button>
        <button
          onClick={() => editor.chain().focus().setTextAlign('justify').run()}
          className={`p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 ${
            editor.isActive({ textAlign: 'justify' }) ? 'bg-gray-200 dark:bg-gray-700' : ''
          }`}
        >
          <FiAlignJustify className="w-4 h-4" />
        </button>
      </div>

      {/* Color Controls */}
      <div className="flex items-center gap-1 border-r border-gray-200 dark:border-gray-700 pr-2">
        <div className="relative group">
          <button
            className="p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
          >
            <FiType className="w-4 h-4" />
          </button>
          <input
            type="color"
            onChange={e => editor.chain().focus().setColor(e.target.value).run()}
            className="absolute top-full left-0 mt-1 opacity-0 w-8 h-8 cursor-pointer"
          />
        </div>
        <div className="relative group">
          <button
            className="p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
          >
            <FiDroplet className="w-4 h-4" />
          </button>
          <input
            type="color"
            onChange={e => {
              editor.chain().focus().setMark('highlight', { color: e.target.value }).run();
            }}
            className="absolute top-full left-0 mt-1 opacity-0 w-8 h-8 cursor-pointer"
          />
        </div>
      </div>

      {/* Link and Image Controls */}
      <div className="flex items-center gap-1 border-r border-gray-200 dark:border-gray-700 pr-2">
        <button
          onClick={setLink}
          className={`p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 ${
            editor.isActive('link') ? 'bg-gray-200 dark:bg-gray-700' : ''
          }`}
        >
          <FiLink className="w-4 h-4" />
        </button>
        <button
          onClick={addImage}
          className="p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
        >
          <FiImage className="w-4 h-4" />
        </button>
      </div>

      {/* Undo/Redo Controls */}
      <div className="flex items-center gap-1">
        <button
          onClick={() => editor.chain().focus().undo().run()}
          className="p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
        >
          <FiRotateCcw className="w-4 h-4" />
        </button>
        <button
          onClick={() => editor.chain().focus().redo().run()}
          className="p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
        >
          <FiRotateCw className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

const RichTextEditor = ({ content, onChange, dir = 'ltr', placeholder = 'Start writing...', type, metadata }) => {
  const editor = useEditor({
    extensions: [
      StarterKit.configure({
        heading: type === 'heading'
      }),
      Image,
      TextAlign.configure({
        types: ['heading', 'paragraph'],
      }),
      Link.configure({
        openOnClick: false,
      }),
      Placeholder.configure({
        placeholder,
      }),
      TextStyle,
      Color,
    ],
    content,
    onUpdate: ({ editor }) => {
      onChange(editor.getHTML());
    },
  });

  return (
    <div className="border rounded-lg overflow-hidden bg-white dark:bg-slate-900 text-gray-900 dark:text-gray-100">
      <MenuBar editor={editor} type={type} metadata={metadata} />
      <div 
        className="p-4 min-h-[200px] prose dark:prose-invert max-w-none" 
        dir={dir}
        style={{
          color: metadata?.style?.textColor || 'inherit',
          backgroundColor: metadata?.style?.backgroundColor || 'transparent',
          marginTop: metadata?.style?.margins?.top ? `${metadata.style.margins.top}px` : 0,
          marginBottom: metadata?.style?.margins?.bottom ? `${metadata.style.margins.bottom}px` : 0,
        }}
      >
        <EditorContent editor={editor} />
      </div>
    </div>
  );
};

export default RichTextEditor; 