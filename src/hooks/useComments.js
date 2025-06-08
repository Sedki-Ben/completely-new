import { useState, useEffect } from 'react';
import { comments as commentApi } from '../services/api';
import { useAuth } from '../contexts/AuthContext';

export const useComments = (articleId, onCommentCountChange) => {
    const [comments, setComments] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const { user } = useAuth();

    // Fetch comments for article
    const fetchComments = async (pageNum = 1, limit = 10) => {
        try {
            setLoading(true);
            setError(null);
            const response = await commentApi.getByArticle(articleId, { page: pageNum, limit });
            
            if (pageNum === 1) {
                setComments(response.data.comments);
            } else {
                setComments(prev => [...prev, ...response.data.comments]);
            }
            
            setTotalPages(response.data.totalPages);
            setPage(pageNum);
        } catch (err) {
            setError(err.response?.data?.message || 'Failed to fetch comments');
        } finally {
            setLoading(false);
        }
    };

    // Create a new comment
    const createComment = async (content, parentId = null) => {
        try {
            setError(null);
            const response = await commentApi.create(articleId, { content, parentId });
            
            if (parentId) {
                // If it's a reply, add to the parent comment's replies
                setComments(prev => prev.map(comment => {
                    if (comment._id === parentId) {
                        return {
                            ...comment,
                            replies: [...(comment.replies || []), response.data]
                        };
                    }
                    return comment;
                }));
            } else {
                // If it's a top-level comment, add to the beginning
                setComments(prev => [response.data, ...prev]);
            }
            
            return response.data;
        } catch (err) {
            setError(err.response?.data?.message || 'Failed to create comment');
            throw err;
        }
    };

    // Update a comment
    const updateComment = async (commentId, content) => {
        try {
            setError(null);
            const response = await commentApi.update(commentId, { content });
            
            // Update comment in state
            setComments(prev => prev.map(comment => {
                if (comment._id === commentId) {
                    return response.data;
                }
                // Check replies
                if (comment.replies) {
                    return {
                        ...comment,
                        replies: comment.replies.map(reply => 
                            reply._id === commentId ? response.data : reply
                        )
                    };
                }
                return comment;
            }));
            
            return response.data;
        } catch (err) {
            setError(err.response?.data?.message || 'Failed to update comment');
            throw err;
        }
    };

    // Delete a comment
    const deleteComment = async (commentId) => {
        try {
            setError(null);
            await commentApi.delete(commentId);
            
            // Remove comment from state
            setComments(prev => prev.filter(comment => {
                if (comment._id === commentId) {
                    return false;
                }
                // Also filter from replies
                if (comment.replies) {
                    comment.replies = comment.replies.filter(reply => reply._id !== commentId);
                }
                return true;
            }));
        } catch (err) {
            setError(err.response?.data?.message || 'Failed to delete comment');
            throw err;
        }
    };

    // Toggle like on comment
    const toggleCommentLike = async (commentId) => {
        try {
            setError(null);
            const response = await commentApi.toggleLike(commentId);
            
            // Update like status in state
            setComments(prev => prev.map(comment => {
                if (comment._id === commentId) {
                    return {
                        ...comment,
                        likes: {
                            count: response.data.likes,
                            users: response.data.hasLiked 
                                ? [...(comment.likes?.users || []), user._id]
                                : (comment.likes?.users || []).filter(id => id !== user._id)
                        }
                    };
                }
                // Check replies
                if (comment.replies) {
                    return {
                        ...comment,
                        replies: comment.replies.map(reply => 
                            reply._id === commentId ? {
                                ...reply,
                                likes: {
                                    count: response.data.likes,
                                    users: response.data.hasLiked 
                                        ? [...(reply.likes?.users || []), user._id]
                                        : (reply.likes?.users || []).filter(id => id !== user._id)
                                }
                            } : reply
                        )
                    };
                }
                return comment;
            }));
            
            return response.data;
        } catch (err) {
            setError(err.response?.data?.message || 'Failed to toggle comment like');
            throw err;
        }
    };

    // Report a comment
    const reportComment = async (commentId, reason) => {
        try {
            setError(null);
            await commentApi.report(commentId, { reason });
        } catch (err) {
            setError(err.response?.data?.message || 'Failed to report comment');
            throw err;
        }
    };

    // Load more comments (pagination)
    const loadMoreComments = () => {
        if (page < totalPages) {
            fetchComments(page + 1);
        }
    };

    // Initialize comments when articleId changes
    useEffect(() => {
        if (articleId) {
            fetchComments();
        }
    }, [articleId]);

    // Calculate total comment count (including replies) and notify parent
    useEffect(() => {
        const totalCount = comments.reduce((total, comment) => {
            return total + 1 + (comment.replies ? comment.replies.length : 0);
        }, 0);
        
        if (onCommentCountChange) {
            onCommentCountChange(totalCount);
        }
    }, [comments, onCommentCountChange]);

    return {
        comments,
        loading,
        error,
        page,
        totalPages,
        fetchComments,
        createComment,
        updateComment,
        deleteComment,
        toggleCommentLike,
        reportComment,
        loadMoreComments,
        hasMoreComments: page < totalPages
    };
}; 