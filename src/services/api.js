const API_URL = process.env.REACT_APP_API_URL || '';

// Helper function to handle API responses
const handleResponse = async (response) => {
  if (response.status === 429) {
    const data = await response.json();
    throw new Error(data.message || 'Too many requests. Please try again later.');
  }
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.message || 'Something went wrong');
  }
  
  return response.json();
};

// Helper function to get auth header
const getAuthHeader = () => {
  const token = localStorage.getItem('token');
  return token ? { 'Authorization': `Bearer ${token}` } : {};
};

const api = {
  auth: {
    login: async (email, password) => {
      const response = await fetch(`${API_URL}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      const data = await handleResponse(response);
      if (data.token) {
        localStorage.setItem('token', data.token);
      }
      return data;
    },

    register: async (userData) => {
      const response = await fetch(`${API_URL}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(userData)
      });
      return handleResponse(response);
    },

    logout: () => {
      localStorage.removeItem('token');
    }
  },

  articles: {
    getAll: async (page = 1, limit = 10) => {
      const response = await fetch(`${API_URL}/api/articles?page=${page}&limit=${limit}`);
      return handleResponse(response);
    },

    getOne: async (id) => {
      const response = await fetch(`${API_URL}/api/articles/${id}`);
      return handleResponse(response);
    },

    create: async (articleData) => {
      const response = await fetch(`${API_URL}/api/articles`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeader()
        },
        body: JSON.stringify(articleData)
      });
      return handleResponse(response);
    },

    like: async (articleId) => {
      const response = await fetch(`${API_URL}/api/articles/${articleId}/like`, {
        method: 'POST',
        headers: getAuthHeader()
      });
      return handleResponse(response);
    }
  },

  comments: {
    add: async (articleId, comment) => {
      const response = await fetch(`${API_URL}/api/articles/${articleId}/comments`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeader()
        },
        body: JSON.stringify(comment)
      });
      return handleResponse(response);
    },

    reply: async (articleId, commentId, reply) => {
      const response = await fetch(`${API_URL}/api/articles/${articleId}/comments/${commentId}/replies`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeader()
        },
        body: JSON.stringify(reply)
      });
      return handleResponse(response);
    }
  },

  newsletter: {
    subscribe: async (email, preferences) => {
      const response = await fetch(`${API_URL}/api/newsletter/subscribe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, preferences })
      });
      return handleResponse(response);
    },

    unsubscribe: async (token) => {
      const response = await fetch(`${API_URL}/api/newsletter/unsubscribe/${token}`);
      return handleResponse(response);
    },

    updatePreferences: async (preferences) => {
      const response = await fetch(`${API_URL}/api/newsletter/preferences`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeader()
        },
        body: JSON.stringify(preferences)
      });
      return handleResponse(response);
    }
  },

  user: {
    updateLanguage: async (language) => {
      const response = await fetch(`${API_URL}/api/users/language`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          ...getAuthHeader()
        },
        body: JSON.stringify({ language })
      });
      return handleResponse(response);
    },

    getProfile: async () => {
      const response = await fetch(`${API_URL}/api/users/profile`, {
        headers: getAuthHeader()
      });
      return handleResponse(response);
    }
  }
};

export default api; 
 