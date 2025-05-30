import axios from 'axios';

const api = axios.create({
    baseURL: process.env.REACT_APP_API_URL || '/api',
    headers: {
        'Content-Type': 'application/json'
    }
});

// Add a request interceptor to add auth token
api.interceptors.request.use(
    (config) => {
  const token = localStorage.getItem('token');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// Add a response interceptor to handle errors
api.interceptors.response.use(
    (response) => response,
    (error) => {
        // Handle 401 Unauthorized errors
        if (error.response?.status === 401) {
            localStorage.removeItem('token');
            // Optionally redirect to login page or trigger auth modal
            window.location.href = '/login';
        }
        return Promise.reject(error);
    }
);

// Auth endpoints
export const auth = {
    login: (email, password) => api.post('/auth/login', { email, password }),
    register: (userData) => api.post('/auth/register', userData),
    logout: () => api.post('/auth/logout'),
    getCurrentUser: () => api.get('/auth/me'),
    updateProfile: (data) => api.put('/auth/profile', data),
    changePassword: (currentPassword, newPassword) => api.put('/auth/password', { currentPassword, newPassword }),
    forgotPassword: (email) => api.post('/auth/forgot-password', { email }),
    resetPassword: (token, password) => api.post(`/auth/reset-password/${token}`, { password })
};

// Article endpoints
export const articles = {
    getAll: (params) => api.get('/articles', { params }),
    getByType: (type, params) => api.get(`/articles/type/${type}`, { params }),
    getBySlug: (slug) => api.get(`/articles/slug/${slug}`),
    create: (data) => {
        if (data instanceof FormData) {
            return api.post('/articles', data, { headers: { 'Content-Type': 'multipart/form-data' } });
        }
        return api.post('/articles', data);
    },
    update: (id, data) => {
        if (data instanceof FormData) {
            return api.put(`/articles/${id}`, data, { headers: { 'Content-Type': 'multipart/form-data' } });
        }
        return api.put(`/articles/${id}`, data);
    },
    delete: (id) => api.delete(`/articles/${id}`),
    toggleLike: (id) => api.post(`/articles/${id}/like`),
    share: (id, platform) => api.post(`/articles/${id}/share`, { platform }),
    getStats: () => api.get('/articles/stats/me'),
    getDrafts: () => api.get('/articles/drafts/me'),
    publish: (id) => api.post(`/articles/${id}/publish`),
    archive: (id) => api.post(`/articles/${id}/archive`)
};

// Comment endpoints
export const comments = {
    getArticleComments: (articleId) => api.get(`/comments/article/${articleId}`),
    getReplies: (commentId) => api.get(`/comments/${commentId}/replies`),
    create: (data) => api.post('/comments', data),
    update: (id, content) => api.put(`/comments/${id}`, { content }),
    delete: (id) => api.delete(`/comments/${id}`),
    toggleLike: (id) => api.post(`/comments/${id}/like`),
    report: (id, reason) => api.post(`/comments/${id}/report`, { reason })
};

// Newsletter endpoints
export const newsletter = {
    subscribe: (email, preferences) => api.post('/newsletter/subscribe', { email, preferences }),
    unsubscribe: (email) => api.post('/newsletter/unsubscribe', { email }),
    updatePreferences: (email, preferences) => api.put('/newsletter/preferences', { email, preferences }),
    verifySubscription: (token) => api.get(`/newsletter/verify/${token}`),
    // Admin endpoints
    getSubscribers: () => api.get('/newsletter/subscribers'),
    sendNewsletter: (data) => api.post('/newsletter/send', data),
    getStats: () => api.get('/newsletter/stats'),
    deleteSubscriber: (email) => api.delete(`/newsletter/subscribers/${email}`)
};

export default api; 
 