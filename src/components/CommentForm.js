import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../contexts/AuthContext';
import { getUserAvatarUrl } from '../utils/imageUtils';
import defaultMaleAvatar from '../assets/images/mann.png';
import defaultFemaleAvatar from '../assets/images/frau.png';

const CommentForm = ({ 
    onSubmit, 
    onCancel, 
    placeholder = '', 
    submitText = 'Post Comment',
    initialValue = '',
    showAvatar = true,
    theme
}) => {
    const { t, i18n } = useTranslation();
    const { user } = useAuth();
    const [content, setContent] = useState(initialValue);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const isRTL = i18n.language === 'ar';

    // Get theme colors or default
    const themeColors = theme || {
        light: 'bg-gray-100 text-gray-900 dark:bg-gray-900 dark:text-gray-100',
        border: 'border-gray-900 dark:border-gray-600',
        hover: 'hover:text-gray-900 dark:hover:text-gray-400',
        icon: 'text-gray-500'
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        
        if (!content.trim()) {
            setError(t('Comment cannot be empty'));
            return;
        }

        if (content.length > 1000) {
            setError(t('Comment is too long (max 1000 characters)'));
            return;
        }

        setLoading(true);
        setError('');

        try {
            await onSubmit(content.trim());
            setContent('');
        } catch (error) {
            setError(error.message || t('Failed to post comment'));
        } finally {
            setLoading(false);
        }
    };

    const handleCancel = () => {
        setContent(initialValue);
        setError('');
        if (onCancel) {
            onCancel();
        }
    };

    return (
        <form onSubmit={handleSubmit} className="space-y-4">
            <div className={`flex gap-3 ${isRTL ? 'flex-row-reverse' : ''}`}>
                {showAvatar && user && (
                    <img
                        src={getUserAvatarUrl(user)}
                        alt={user.name}
                        className="w-10 h-10 rounded-full object-cover flex-shrink-0 mt-1"
                        onError={(e) => {
                            e.target.src = user.gender === 'female' ? defaultFemaleAvatar : defaultMaleAvatar;
                        }}
                    />
                )}
                <div className="flex-1">
                    <textarea
                        value={content}
                        onChange={(e) => setContent(e.target.value)}
                        placeholder={placeholder || t('Write a comment...')}
                        className={`w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg 
                            bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                            focus:ring-2 focus:ring-blue-500 focus:border-transparent
                            resize-none transition-colors ${isRTL ? 'text-right' : 'text-left'}`}
                        rows={3}
                        maxLength={1000}
                        disabled={loading}
                        dir={isRTL ? 'rtl' : 'ltr'}
                    />
                    
                    {/* Character count */}
                    <div className={`mt-1 text-xs text-gray-500 ${isRTL ? 'text-left' : 'text-right'}`}>
                        {content.length}/1000
                    </div>

                    {/* Error message */}
                    {error && (
                        <div className="mt-2 text-sm text-red-600 dark:text-red-400">
                            {error}
                        </div>
                    )}

                    {/* Form actions */}
                    <div className={`mt-3 flex gap-2 ${isRTL ? 'flex-row-reverse' : ''}`}>
                        <button
                            type="submit"
                            disabled={loading || !content.trim()}
                            className={`px-4 py-2 ${themeColors.light} rounded-lg ${themeColors.hover}
                                disabled:opacity-50 disabled:cursor-not-allowed
                                focus:ring-2 focus:ring-offset-2
                                transition-colors ${loading ? 'opacity-50' : ''}`}
                        >
                            {loading ? t('Posting...') : t(submitText)}
                        </button>
                        
                        {onCancel && (
                            <button
                                type="button"
                                onClick={handleCancel}
                                disabled={loading}
                                className="px-4 py-2 bg-gray-200 dark:bg-gray-600 text-gray-800 dark:text-gray-200 
                                    rounded-lg hover:bg-gray-300 dark:hover:bg-gray-500
                                    focus:ring-2 focus:ring-gray-500 focus:ring-offset-2
                                    transition-colors disabled:opacity-50"
                            >
                                {t('Cancel')}
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </form>
    );
};

export default CommentForm; 