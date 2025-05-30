import React, { createContext, useState, useContext } from 'react';
import api from '../services/api';

const CommentContext = createContext(null);

export const useComments = () => {
    const context = useContext(CommentContext);
    if (!context) {
        throw new Error('useComments must be used within a CommentProvider');
    }
    return context;
};

export const CommentProvider = ({ children }) => {
    const [comments, setComments] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const fetchArticleComments = async (articleId) => {
        try {
            setLoading(true);
            setError(null);
            const response = await api.get(`/comments/article/${articleId}`);
            setComments(response.data);
            return response.data;
        } catch (err) {
            setError(err.response?.data?.msg || 'Failed to fetch comments');
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const fetchCommentReplies = async (commentId) => {
        try {
            setLoading(true);
            setError(null);
            const response = await api.get(`/comments/${commentId}/replies`);
            return response.data;
        } catch (err) {
            setError(err.response?.data?.msg || 'Failed to fetch replies');
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const createComment = async (data) => {
        try {
            setLoading(true);
            setError(null);
            const response = await api.post('/comments', data);
            setComments(prev => [response.data, ...prev]);
            return response.data;
        } catch (err) {
            setError(err.response?.data?.msg || 'Failed to create comment');
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const updateComment = async (id, content) => {
        try {
            setLoading(true);
            setError(null);
            const response = await api.put(`/comments/${id}`, { content });
            setComments(prev => prev.map(comment => 
                comment._id === id ? response.data : comment
            ));
            return response.data;
        } catch (err) {
            setError(err.response?.data?.msg || 'Failed to update comment');
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const deleteComment = async (id) => {
        try {
            setLoading(true);
            setError(null);
            await api.delete(`/comments/${id}`);
            setComments(prev => prev.filter(comment => comment._id !== id));
        } catch (err) {
            setError(err.response?.data?.msg || 'Failed to delete comment');
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const toggleLike = async (id) => {
        try {
            setError(null);
            const response = await api.post(`/comments/${id}/like`);
            setComments(prev => prev.map(comment => 
                comment._id === id ? {
                    ...comment,
                    likes: response.data.likes
                } : comment
            ));
            return response.data;
        } catch (err) {
            setError(err.response?.data?.msg || 'Failed to toggle like');
            throw err;
        }
    };

    const reportComment = async (id, reason) => {
        try {
            setError(null);
            await api.post(`/comments/${id}/report`, { reason });
        } catch (err) {
            setError(err.response?.data?.msg || 'Failed to report comment');
            throw err;
        }
    };

    const value = {
        comments,
        loading,
        error,
        fetchArticleComments,
        fetchCommentReplies,
        createComment,
        updateComment,
        deleteComment,
        toggleLike,
        reportComment
    };

    return (
        <CommentContext.Provider value={value}>
            {children}
        </CommentContext.Provider>
    );
};

export default CommentContext; 