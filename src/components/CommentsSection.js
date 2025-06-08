import React from 'react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../contexts/AuthContext';
import { useComments } from '../hooks/useComments';
import Comment from './Comment';
import CommentForm from './CommentForm';
import { FiMessageCircle, FiUser } from 'react-icons/fi';
import { Link } from 'react-router-dom';

const CommentsSection = ({ articleId, category, theme, onCommentCountChange }) => {
    const { t, i18n } = useTranslation();
    const { isAuthenticated } = useAuth();
    const {
        comments,
        loading,
        error,
        createComment,
        updateComment,
        deleteComment,
        toggleCommentLike,
        reportComment,
        loadMoreComments,
        hasMoreComments
    } = useComments(articleId, onCommentCountChange);

    const isRTL = i18n.language === 'ar';

    // Get theme colors or default
    const themeColors = theme || {
        light: 'bg-gray-100 text-gray-900 dark:bg-gray-900 dark:text-gray-100',
        border: 'border-gray-900 dark:border-gray-600',
        hover: 'hover:text-gray-900 dark:hover:text-gray-400',
        icon: 'text-gray-500'
    };

    const handleCreateComment = async (content) => {
        return await createComment(content);
    };

    const handleReplyToComment = async (content, parentId) => {
        return await createComment(content, parentId);
    };

    if (loading && comments.length === 0) {
        return (
            <div className="max-w-4xl mx-auto mt-12 p-6">
                <div className="flex items-center justify-center py-8">
                    <div className={`animate-spin rounded-full h-8 w-8 border-b-2 ${themeColors.icon.replace('text-', 'border-')}`}></div>
                    <span className="ml-3 text-gray-600 dark:text-gray-400">{t('Loading comments...')}</span>
                </div>
            </div>
        );
    }

    return (
        <div className="max-w-4xl mx-auto mt-12 p-6 bg-white dark:bg-gray-800 rounded-xl shadow-lg border-l-4" style={{borderLeftColor: themeColors.icon.includes('red') ? '#ef4444' : themeColors.icon.includes('green') ? '#22c55e' : themeColors.icon.includes('purple') ? '#a855f7' : themeColors.icon.includes('yellow') ? '#eab308' : '#6b7280'}}>
            {/* Comments Header */}
            <div className={`flex items-center gap-3 mb-6 pb-4 border-b ${themeColors.border} ${isRTL ? 'flex-row-reverse' : ''}`}>
                <FiMessageCircle className={`w-6 h-6 ${themeColors.icon}`} />
                <h3 className={`text-2xl font-bold ${themeColors.icon}`}>
                    {t('Comments')} ({comments.length})
                </h3>
            </div>

            {/* Error Display */}
            {error && (
                <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                    <p className="text-red-600 dark:text-red-400">{error}</p>
                </div>
            )}

            {/* Comment Form for Authenticated Users */}
            {isAuthenticated ? (
                <div className="mb-8">
                    <CommentForm
                        onSubmit={handleCreateComment}
                        placeholder={t('Share your thoughts...')}
                        submitText={t('Post Comment')}
                        theme={themeColors}
                    />
                </div>
            ) : (
                <div className={`mb-8 p-6 bg-gray-50 dark:bg-gray-700 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-600 ${isRTL ? 'text-right' : 'text-left'}`}>
                    <div className={`flex items-center gap-3 ${isRTL ? 'flex-row-reverse' : ''}`}>
                        <FiUser className="w-5 h-5 text-gray-500" />
                        <p className="text-gray-600 dark:text-gray-400">
                            {t('Please')} {' '}
                            <Link 
                                to="/signin" 
                                className={`${themeColors.icon} ${themeColors.hover} hover:underline font-medium`}
                            >
                                {t('sign in')}
                            </Link>
                            {' '} {t('to join the conversation')}
                        </p>
                    </div>
                </div>
            )}

            {/* Comments List */}
            <div className="space-y-6">
                {comments.length === 0 ? (
                    <div className={`text-center py-12 ${isRTL ? 'text-right' : 'text-left'}`}>
                        <FiMessageCircle className="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
                        <p className="text-gray-500 dark:text-gray-400 text-lg">
                            {t('No comments yet')}
                        </p>
                        <p className="text-gray-400 dark:text-gray-500 mt-2">
                            {t('Be the first to share your thoughts!')}
                        </p>
                    </div>
                ) : (
                    <>
                        {comments.map(comment => (
                            <Comment
                                key={comment._id}
                                comment={comment}
                                onUpdate={updateComment}
                                onDelete={deleteComment}
                                onLike={toggleCommentLike}
                                onReply={handleReplyToComment}
                                onReport={reportComment}
                                theme={themeColors}
                            />
                        ))}

                        {/* Load More Button */}
                        {hasMoreComments && (
                            <div className="flex justify-center pt-6">
                                <button
                                    onClick={loadMoreComments}
                                    disabled={loading}
                                    className={`px-6 py-3 ${themeColors.light} rounded-lg ${themeColors.hover}
                                        disabled:opacity-50 disabled:cursor-not-allowed
                                        focus:ring-2 focus:ring-offset-2
                                        transition-colors flex items-center gap-2`}
                                >
                                    {loading ? (
                                        <>
                                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                                            {t('Loading...')}
                                        </>
                                    ) : (
                                        t('Load More Comments')
                                    )}
                                </button>
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
};

export default CommentsSection; 
 