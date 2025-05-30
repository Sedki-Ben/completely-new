import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../contexts/AuthContext';
import { useComments } from '../contexts/CommentContext';
import { formatDistanceToNow } from 'date-fns';

const Comment = ({ comment, articleId, onReply }) => {
    const { t } = useTranslation();
    const { user } = useAuth();
    const { updateComment, deleteComment, toggleLike, reportComment } = useComments();
    
    const [isEditing, setIsEditing] = useState(false);
    const [editContent, setEditContent] = useState(comment.content);
    const [showReplyForm, setShowReplyForm] = useState(false);
    const [replyContent, setReplyContent] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const isAuthor = user && comment.author._id === user._id;
    const hasLiked = user && comment.likes.users.includes(user._id);

    const handleEdit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        try {
            await updateComment(comment._id, editContent);
            setIsEditing(false);
        } catch (err) {
            setError(t('Failed to update comment'));
        } finally {
            setLoading(false);
        }
    };

    const handleDelete = async () => {
        if (!window.confirm(t('Are you sure you want to delete this comment?'))) {
            return;
        }

        setLoading(true);
        setError('');

        try {
            await deleteComment(comment._id);
        } catch (err) {
            setError(t('Failed to delete comment'));
        } finally {
            setLoading(false);
        }
    };

    const handleLike = async () => {
        if (!user) {
            // Trigger auth modal through parent component
            return;
        }

        try {
            await toggleLike(comment._id);
        } catch (err) {
            setError(t('Failed to like comment'));
        }
    };

    const handleReport = async () => {
        if (!user) {
            // Trigger auth modal through parent component
            return;
        }

        const reason = window.prompt(t('Please provide a reason for reporting this comment:'));
        if (!reason) return;

        try {
            await reportComment(comment._id, reason);
            alert(t('Comment reported successfully'));
        } catch (err) {
            setError(t('Failed to report comment'));
        }
    };

    const handleReply = async (e) => {
        e.preventDefault();
        if (!user) {
            // Trigger auth modal through parent component
            return;
        }

        setLoading(true);
        setError('');

        try {
            await onReply({
                content: replyContent,
                articleId,
                parentComment: comment._id
            });
            setReplyContent('');
            setShowReplyForm(false);
        } catch (err) {
            setError(t('Failed to post reply'));
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="border-l-2 border-gray-200 pl-4 mb-4">
            <div className="flex items-start space-x-3">
                <img
                    src={comment.author.profilePicture || '/default-avatar.png'}
                    alt={comment.author.name}
                    className="w-8 h-8 rounded-full"
                />
                <div className="flex-1">
                    <div className="flex items-center space-x-2">
                        <span className="font-medium text-gray-900 dark:text-white">
                            {comment.author.name}
                        </span>
                        <span className="text-sm text-gray-500">
                            {formatDistanceToNow(new Date(comment.createdAt), { addSuffix: true })}
                        </span>
                        {comment.isEdited && (
                            <span className="text-sm text-gray-500">{t('(edited)')}</span>
                        )}
                    </div>

                    {isEditing ? (
                        <form onSubmit={handleEdit} className="mt-2">
                            <textarea
                                value={editContent}
                                onChange={(e) => setEditContent(e.target.value)}
                                className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                                rows={3}
                                required
                            />
                            <div className="mt-2 space-x-2">
                                <button
                                    type="submit"
                                    disabled={loading}
                                    className="px-3 py-1 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700"
                                >
                                    {loading ? t('Saving...') : t('Save')}
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setIsEditing(false)}
                                    className="px-3 py-1 text-sm bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 dark:bg-gray-600 dark:text-gray-200 dark:hover:bg-gray-500"
                                >
                                    {t('Cancel')}
                                </button>
                            </div>
                        </form>
                    ) : (
                        <p className="mt-1 text-gray-800 dark:text-gray-200">
                            {comment.content}
                        </p>
                    )}

                    {error && (
                        <p className="mt-2 text-sm text-red-500">{error}</p>
                    )}

                    <div className="mt-2 flex items-center space-x-4 text-sm">
                        <button
                            onClick={handleLike}
                            className={`flex items-center space-x-1 ${
                                hasLiked ? 'text-blue-600' : 'text-gray-500 hover:text-blue-600'
                            }`}
                        >
                            <svg className="w-4 h-4" fill={hasLiked ? 'currentColor' : 'none'} stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
                            </svg>
                            <span>{comment.likes.count}</span>
                        </button>

                        <button
                            onClick={() => setShowReplyForm(!showReplyForm)}
                            className="text-gray-500 hover:text-blue-600"
                        >
                            {t('Reply')}
                        </button>

                        {isAuthor && (
                            <>
                                <button
                                    onClick={() => setIsEditing(true)}
                                    className="text-gray-500 hover:text-blue-600"
                                >
                                    {t('Edit')}
                                </button>
                                <button
                                    onClick={handleDelete}
                                    className="text-gray-500 hover:text-red-600"
                                >
                                    {t('Delete')}
                                </button>
                            </>
                        )}

                        {!isAuthor && user && (
                            <button
                                onClick={handleReport}
                                className="text-gray-500 hover:text-red-600"
                            >
                                {t('Report')}
                            </button>
                        )}
                    </div>

                    {showReplyForm && (
                        <form onSubmit={handleReply} className="mt-4">
                            <textarea
                                value={replyContent}
                                onChange={(e) => setReplyContent(e.target.value)}
                                placeholder={t('Write a reply...')}
                                className="w-full p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                                rows={2}
                                required
                            />
                            <div className="mt-2 space-x-2">
                                <button
                                    type="submit"
                                    disabled={loading}
                                    className="px-3 py-1 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700"
                                >
                                    {loading ? t('Posting...') : t('Post Reply')}
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setShowReplyForm(false)}
                                    className="px-3 py-1 text-sm bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 dark:bg-gray-600 dark:text-gray-200 dark:hover:bg-gray-500"
                                >
                                    {t('Cancel')}
                                </button>
                            </div>
                        </form>
                    )}
                </div>
            </div>
        </div>
    );
};

export default Comment; 