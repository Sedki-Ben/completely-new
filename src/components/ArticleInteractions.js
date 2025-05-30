import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../contexts/AuthContext';
import { useArticles } from '../contexts/ArticleContext';

const ArticleInteractions = ({ article, onAuthRequired }) => {
    const { t } = useTranslation();
    const { user } = useAuth();
    const { toggleLike, shareArticle } = useArticles();
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const hasLiked = user && article.likes.users.includes(user._id);

    const handleLike = async () => {
        if (!user) {
            onAuthRequired();
            return;
        }

        try {
            setLoading(true);
            setError('');
            await toggleLike(article._id);
        } catch (err) {
            setError(t('Failed to like article'));
        } finally {
            setLoading(false);
        }
    };

    const handleShare = async (platform) => {
        try {
            setLoading(true);
            setError('');

            // Record share in backend
            await shareArticle(article._id, platform);

            // Prepare share URL and text
            const url = window.location.href;
            const text = article.title;
            
            // Open share dialog based on platform
            switch (platform) {
                case 'twitter':
                    window.open(
                        `https://twitter.com/intent/tweet?url=${encodeURIComponent(url)}&text=${encodeURIComponent(text)}`,
                        '_blank'
                    );
                    break;
                case 'facebook':
                    window.open(
                        `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}`,
                        '_blank'
                    );
                    break;
                case 'linkedin':
                    window.open(
                        `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(url)}`,
                        '_blank'
                    );
                    break;
                default:
                    // Copy link to clipboard
                    await navigator.clipboard.writeText(url);
                    alert(t('Link copied to clipboard!'));
            }
        } catch (err) {
            setError(t('Failed to share article'));
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col space-y-4 sm:flex-row sm:space-y-0 sm:space-x-6 items-center">
            <button
                onClick={handleLike}
                disabled={loading}
                className={`flex items-center space-x-2 px-4 py-2 rounded-full transition-colors ${
                    hasLiked
                        ? 'bg-blue-100 text-blue-600 dark:bg-blue-900/30 dark:text-blue-400'
                        : 'bg-gray-100 text-gray-600 hover:bg-blue-100 hover:text-blue-600 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-blue-900/30 dark:hover:text-blue-400'
                }`}
            >
                <svg className="w-5 h-5" fill={hasLiked ? 'currentColor' : 'none'} stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
                </svg>
                <span>{article.likes.count}</span>
            </button>

            <div className="flex items-center space-x-2">
                <button
                    onClick={() => handleShare('twitter')}
                    className="p-2 text-gray-600 hover:text-blue-400 dark:text-gray-400 dark:hover:text-blue-300"
                    title={t('Share on Twitter')}
                >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z"/>
                    </svg>
                </button>

                <button
                    onClick={() => handleShare('facebook')}
                    className="p-2 text-gray-600 hover:text-blue-600 dark:text-gray-400 dark:hover:text-blue-500"
                    title={t('Share on Facebook')}
                >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/>
                    </svg>
                </button>

                <button
                    onClick={() => handleShare('linkedin')}
                    className="p-2 text-gray-600 hover:text-blue-500 dark:text-gray-400 dark:hover:text-blue-400"
                    title={t('Share on LinkedIn')}
                >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                    </svg>
                </button>

                <button
                    onClick={() => handleShare('copy')}
                    className="p-2 text-gray-600 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200"
                    title={t('Copy link')}
                >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                    </svg>
                </button>
            </div>

            {error && (
                <p className="text-sm text-red-500">{error}</p>
            )}

            <div className="text-sm text-gray-500">
                {t('{{count}} shares', { count: article.shares.count })}
            </div>
        </div>
    );
};

export default ArticleInteractions; 