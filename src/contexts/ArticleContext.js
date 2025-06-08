import React, { createContext, useState, useContext } from 'react';
import api from '../services/api';

const ArticleContext = createContext(null);

export const useArticles = () => {
    const context = useContext(ArticleContext);
    if (!context) {
        throw new Error('useArticles must be used within an ArticleProvider');
    }
    return context;
};

export const ArticleProvider = ({ children }) => {
    const [articles, setArticles] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const fetchArticles = async (type) => {
        try {
            setLoading(true);
            setError(null);
            const response = await api.get(type ? `/articles/type/${type}` : '/articles');
            setArticles(response.data);
            return response.data;
        } catch (err) {
            setError(err.response?.data?.msg || 'Failed to fetch articles');
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const getArticleBySlug = async (slug) => {
        try {
            setLoading(true);
            setError(null);
            const response = await api.get(`/articles/slug/${slug}`);
            return response.data;
        } catch (err) {
            setError(err.response?.data?.msg || 'Failed to fetch article');
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const createArticle = async (articleData) => {
        try {
            setLoading(true);
            setError(null);
            const response = await api.post('/articles', articleData);
            setArticles(prev => [response.data, ...prev]);
            return response.data;
        } catch (err) {
            setError(err.response?.data?.msg || 'Failed to create article');
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const updateArticle = async (id, articleData) => {
        try {
            setLoading(true);
            setError(null);
            const response = await api.put(`/articles/${id}`, articleData);
            setArticles(prev => prev.map(article => 
                article._id === id ? response.data : article
            ));
            return response.data;
        } catch (err) {
            setError(err.response?.data?.msg || 'Failed to update article');
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const deleteArticle = async (id) => {
        try {
            setLoading(true);
            setError(null);
            await api.delete(`/articles/${id}`);
            setArticles(prev => prev.filter(article => article._id !== id));
        } catch (err) {
            setError(err.response?.data?.msg || 'Failed to delete article');
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const toggleLike = async (id) => {
        try {
            setError(null);
            const response = await api.post(`/articles/${id}/like`);
            setArticles(prev => prev.map(article => 
                article._id === id ? {
                    ...article,
                    likes: {
                        count: response.data.likes,
                        users: article.likes?.users || []
                    }
                } : article
            ));
            return response.data;
        } catch (err) {
            setError(err.response?.data?.msg || 'Failed to toggle like');
            throw err;
        }
    };

    const shareArticle = async (id, platform) => {
        try {
            setError(null);
            const response = await api.post(`/articles/${id}/share`, { platform });
            setArticles(prev => prev.map(article => 
                article._id === id ? {
                    ...article,
                    shares: response.data.shares
                } : article
            ));
            return response.data;
        } catch (err) {
            setError(err.response?.data?.msg || 'Failed to record share');
            throw err;
        }
    };

    const getWriterStats = async () => {
        try {
            setLoading(true);
            setError(null);
            const response = await api.get('/articles/stats/me');
            return response.data;
        } catch (err) {
            setError(err.response?.data?.msg || 'Failed to fetch writer stats');
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const getWriterDrafts = async () => {
        try {
            setLoading(true);
            setError(null);
            const response = await api.get('/articles/drafts/me');
            return response.data;
        } catch (err) {
            setError(err.response?.data?.msg || 'Failed to fetch drafts');
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const publishArticle = async (id) => {
        try {
            setLoading(true);
            setError(null);
            const response = await api.post(`/articles/${id}/publish`);
            setArticles(prev => prev.map(article => 
                article._id === id ? response.data : article
            ));
            return response.data;
        } catch (err) {
            setError(err.response?.data?.msg || 'Failed to publish article');
            throw err;
        } finally {
            setLoading(false);
        }
    };

    const value = {
        articles,
        loading,
        error,
        fetchArticles,
        getArticleBySlug,
        createArticle,
        updateArticle,
        deleteArticle,
        toggleLike,
        shareArticle,
        getWriterStats,
        getWriterDrafts,
        publishArticle
    };

    return (
        <ArticleContext.Provider value={value}>
            {children}
        </ArticleContext.Provider>
    );
};

export default ArticleContext; 