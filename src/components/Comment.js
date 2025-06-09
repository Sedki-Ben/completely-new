import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../contexts/AuthContext';
import { FiHeart, FiMessageCircle, FiEdit, FiTrash2, FiFlag, FiMoreVertical } from 'react-icons/fi';
import { getUserAvatarUrl } from '../utils/imageUtils';
import CommentForm from './CommentForm';
import defaultMaleAvatar from '../assets/images/mann.png';
import defaultFemaleAvatar from '../assets/images/frau.png';

const Comment = ({ 
    comment, 
    onUpdate, 
    onDelete, 
    onLike, 
    onReply, 
    onReport,
    isReply = false,
    theme
}) => {
    const { t, i18n } = useTranslation();
    const { user, isAuthenticated } = useAuth();
    const [isEditing, setIsEditing] = useState(false);
    const [isReplying, setIsReplying] = useState(false);
    const [showDropdown, setShowDropdown] = useState(false);
    const [showReplies, setShowReplies] = useState(true);

    const isRTL = i18n.language === 'ar';
    const isAuthor = user && comment.author._id === user._id;
    const hasLiked = user && comment.likes?.users?.includes(user._id);
    const isAdmin = user?.role === 'admin';

    // Get theme colors or default
    const themeColors = theme || {
        light: 'bg-gray-100 text-gray-900 dark:bg-gray-900 dark:text-gray-100',
        border: 'border-gray-900 dark:border-gray-600',
        hover: 'hover:text-gray-900 dark:hover:text-gray-400',
        icon: 'text-gray-500'
    };

    const handleEdit = async (content) => {
        try {
            await onUpdate(comment._id, content);
            setIsEditing(false);
        } catch (error) {
            console.error('Failed to edit comment:', error);
        }
    };

    const handleDelete = async () => {
        if (window.confirm(t('Are you sure you want to delete this comment?'))) {
            try {
                await onDelete(comment._id);
            } catch (error) {
                console.error('Failed to delete comment:', error);
            }
        }
    };

    const handleLike = async () => {
        if (!isAuthenticated) {
            // Could trigger auth modal here
            return;
        }
        try {
            await onLike(comment._id);
        } catch (error) {
            console.error('Failed to like comment:', error);
        }
    };

    const handleReply = async (content) => {
        try {
            await onReply(content, comment._id);
            setIsReplying(false);
        } catch (error) {
            console.error('Failed to reply to comment:', error);
        }
    };

    const handleReport = async () => {
        const reason = prompt(t('Please provide a reason for reporting this comment:'));
        if (reason) {
            try {
                await onReport(comment._id, reason);
                setShowDropdown(false);
                alert(t('Comment reported successfully'));
            } catch (error) {
                console.error('Failed to report comment:', error);
            }
        }
    };

    const formatDate = (date) => {
        return new Intl.DateTimeFormat(i18n.language, {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        }).format(new Date(date));
    };

    return (
        <div className={`${isReply ? 'ml-8 md:ml-12' : ''} mb-4`}>
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
                {/* Comment Header */}
                <div className={`flex items-start gap-3 mb-3 ${isRTL ? 'flex-row-reverse' : ''}`}>
                    <img
                        src={getUserAvatarUrl(comment.author)}
                        alt={comment.author.name}
                        className="w-10 h-10 rounded-full object-cover flex-shrink-0"
                        onError={(e) => {
                            e.target.src = comment.author.gender === 'female' ? defaultFemaleAvatar : defaultMaleAvatar;
                        }}
                    />
                    <div className={`flex-1 ${isRTL ? 'text-right' : 'text-left'}`}>
                        <div className={`flex items-center gap-2 ${isRTL ? 'flex-row-reverse' : ''}`}>
                            <h4 className="font-semibold text-gray-900 dark:text-white">
                                {comment.author.name}
                            </h4>
                            <span className="text-sm text-gray-500 dark:text-gray-400">
                                {formatDate(comment.createdAt)}
                            </span>
                            {comment.isEdited && (
                                <span className="text-xs text-gray-400 dark:text-gray-500">
                                    {t('edited')}
                                </span>
                            )}
                        </div>
                    </div>
                    
                    {/* Options Dropdown */}
                    <div className="relative">
                        <button
                            onClick={() => setShowDropdown(!showDropdown)}
                            className="p-1 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                        >
                            <FiMoreVertical className="w-4 h-4 text-gray-500" />
                        </button>
                        
                        {showDropdown && (
                            <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-md shadow-lg border border-gray-200 dark:border-gray-700 z-10">
                                {isAuthor && (
                                    <>
                                        <button
                                            onClick={() => {
                                                setIsEditing(true);
                                                setShowDropdown(false);
                                            }}
                                            className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                                        >
                                            <FiEdit className="w-4 h-4" />
                                            {t('Edit')}
                                        </button>
                                        <button
                                            onClick={handleDelete}
                                            className="flex items-center gap-2 w-full px-4 py-2 text-sm text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20"
                                        >
                                            <FiTrash2 className="w-4 h-4" />
                                            {t('Delete')}
                                        </button>
                                    </>
                                )}
                                {/* Admin can always delete comments */}
                                {!isAuthor && isAdmin && (
                                    <button
                                        onClick={handleDelete}
                                        className="flex items-center gap-2 w-full px-4 py-2 text-sm text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20"
                                    >
                                        <FiTrash2 className="w-4 h-4" />
                                        {t('Delete as Admin')}
                                    </button>
                                )}
                                {!isAuthor && !isAdmin && isAuthenticated && (
                                    <button
                                        onClick={handleReport}
                                        className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                                    >
                                        <FiFlag className="w-4 h-4" />
                                        {t('Report')}
                                    </button>
                                )}
                            </div>
                        )}
                    </div>
                </div>

                {/* Comment Content */}
                <div className={`mb-3 ${isRTL ? 'text-right' : 'text-left'}`}>
                    {isEditing ? (
                        <CommentForm
                            initialValue={comment.content}
                            onSubmit={handleEdit}
                            onCancel={() => setIsEditing(false)}
                            placeholder={t('Edit your comment...')}
                            submitText={t('Save')}
                            theme={themeColors}
                        />
                    ) : (
                        <p className="text-gray-800 dark:text-gray-200 leading-relaxed">
                            {comment.content}
                        </p>
                    )}
                </div>

                {/* Comment Actions */}
                <div className={`flex items-center gap-4 ${isRTL ? 'flex-row-reverse' : ''}`}>
                    <button
                        onClick={handleLike}
                        className={`flex items-center gap-1 px-2 py-1 rounded-lg transition-colors ${
                            hasLiked
                                ? 'text-red-500 bg-red-50 dark:bg-red-900/20'
                                : 'text-gray-500 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20'
                        }`}
                    >
                        <FiHeart className={`w-4 h-4 ${hasLiked ? 'fill-current' : ''}`} />
                        <span className="text-sm">{comment.likes?.count || 0}</span>
                    </button>

                    {!isReply && (
                        <button
                            onClick={() => setIsReplying(!isReplying)}
                            className={`flex items-center gap-1 px-2 py-1 ${themeColors.icon} ${themeColors.hover} rounded-lg transition-colors`}
                        >
                            <FiMessageCircle className="w-4 h-4" />
                            <span className="text-sm">{t('Reply')}</span>
                        </button>
                    )}

                    {comment.replies && comment.replies.length > 0 && (
                        <button
                            onClick={() => setShowReplies(!showReplies)}
                            className={`text-sm ${themeColors.icon} ${themeColors.hover} hover:underline`}
                        >
                            {showReplies ? t('Hide replies') : t('Show replies')} ({comment.replies.length})
                        </button>
                    )}
                </div>

                {/* Reply Form */}
                {isReplying && (
                    <div className="mt-4">
                        <CommentForm
                            onSubmit={handleReply}
                            onCancel={() => setIsReplying(false)}
                            placeholder={t('Write a reply...')}
                            submitText={t('Reply')}
                            theme={themeColors}
                        />
                    </div>
                )}

                {/* Replies */}
                {showReplies && comment.replies && comment.replies.length > 0 && (
                    <div className="mt-4 space-y-3">
                        {comment.replies.map(reply => (
                            <Comment
                                key={reply._id}
                                comment={reply}
                                onUpdate={onUpdate}
                                onDelete={onDelete}
                                onLike={onLike}
                                onReport={onReport}
                                isReply={true}
                                theme={themeColors}
                            />
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

export default Comment; 