import axios from 'axios';

const BASE_URL = '/api/analytics';

export const analytics = {
    // Track article view
    trackView: async (articleId) => {
        try {
            const response = await axios.post(`${BASE_URL}/views/${articleId}`);
            return response.data;
        } catch (error) {
            console.error('Error tracking view:', error);
            throw error;
        }
    },

    // Track article engagement time
    trackEngagement: async (articleId, timeSpentSeconds) => {
        try {
            const response = await axios.post(`${BASE_URL}/engagement/${articleId}`, {
                timeSpent: timeSpentSeconds
            });
            return response.data;
        } catch (error) {
            console.error('Error tracking engagement:', error);
            throw error;
        }
    },

    // Track user interaction (like, share, comment)
    trackInteraction: async (articleId, interactionType) => {
        try {
            const response = await axios.post(`${BASE_URL}/interaction/${articleId}`, {
                type: interactionType
            });
            return response.data;
        } catch (error) {
            console.error('Error tracking interaction:', error);
            throw error;
        }
    },

    // Get article analytics
    getArticleAnalytics: async (articleId) => {
        try {
            const response = await axios.get(`${BASE_URL}/article/${articleId}`);
            return response.data;
        } catch (error) {
            console.error('Error getting analytics:', error);
            throw error;
        }
    },

    // Get writer's analytics dashboard data
    getWriterAnalytics: async () => {
        try {
            const response = await axios.get(`${BASE_URL}/writer/dashboard`);
            return response.data;
        } catch (error) {
            console.error('Error getting writer analytics:', error);
            throw error;
        }
    }
}; 