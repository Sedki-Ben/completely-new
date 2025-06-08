import { useState, useCallback, useEffect } from 'react';
import { articles as articlesAPI } from '../services/api';

// Default author info to maintain consistency
const DEFAULT_AUTHOR = "Sedki B.Haouala";
const DEFAULT_AUTHOR_IMAGE = "/uploads/profile/bild3.jpg"; // Backend path

// Global article cache to share across all hook instances
const globalArticleCache = {
  data: new Map(), // Map of article ID to article data
  subscribers: new Set(), // Set of update callbacks
  
  // Update an article in the cache and notify all subscribers
  updateArticle: (articleId, updates) => {
    const current = globalArticleCache.data.get(articleId);
    console.log('GlobalCache: Updating article', { articleId, updates, currentExists: !!current });
    
    if (current) {
      const updated = { ...current, ...updates };
      globalArticleCache.data.set(articleId, updated);
      
      console.log('GlobalCache: Notifying', globalArticleCache.subscribers.size, 'subscribers');
      // Notify all subscribers
      globalArticleCache.subscribers.forEach(callback => {
        callback(articleId, updated);
      });
    } else {
      console.warn('GlobalCache: Article not found in cache, cannot update:', articleId);
      console.log('GlobalCache: Available articles:', Array.from(globalArticleCache.data.keys()));
    }
  },
  
  // Subscribe to article updates
  subscribe: (callback) => {
    globalArticleCache.subscribers.add(callback);
    return () => globalArticleCache.subscribers.delete(callback);
  },
  
  // Store articles in cache
  cacheArticles: (articles) => {
    articles.forEach(article => {
      if (article.id) {
        globalArticleCache.data.set(article.id, article);
      }
    });
  }
};

// Global function to update comment count from anywhere in the app
export const updateArticleCommentCount = (articleId, newCount) => {
  console.log('updateArticleCommentCount called:', { articleId, newCount });
  globalArticleCache.updateArticle(articleId, { comments: newCount });
};

// Category translations (keeping from original structure)
export const categoryTranslations = {
  'etoile-du-sahel': { 
    en: 'Etoile Du Sahel', 
    fr: 'Étoile Du Sahel', 
    ar: 'النجم الساحلي' 
  },
  'the-beautiful-game': { 
    en: 'The Beautiful Game', 
    fr: 'Le Beau Jeu', 
    ar: 'اللعبة الجميلة' 
  },
  'all-sports-hub': { 
    en: 'All-Sports Hub', 
    fr: 'Centre Omnisports', 
    ar: 'مركز كل الرياضات' 
  },
  archive: { 
    en: 'Archive', 
    fr: 'Archives', 
    ar: 'الأرشيف' 
  },
};

// Transform backend article to match frontend expectations
const transformArticle = (article) => {
  if (!article) return null;
  
  // Get backend URL for constructing full image URLs
  const backendUrl = process.env.REACT_APP_API_URL?.replace('/api', '') || 'http://localhost:5000';
  
  const publishedDate = article.publishedAt || article.createdAt;
  
  return {
    id: article._id,
    _id: article._id, // Keep both for compatibility
    translations: article.translations,
    author: article.author?.name || DEFAULT_AUTHOR,
    authorImage: article.authorImage ? `${backendUrl}${article.authorImage}` : `${backendUrl}${DEFAULT_AUTHOR_IMAGE}`,
    date: new Date(publishedDate).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    }),
    rawDate: publishedDate, // Keep raw date for sorting
    image: article.image ? `${backendUrl}${article.image}` : null,
    category: article.category,
    likes: {
      count: article.likes?.count || 0,
      users: article.likes?.users || []
    },
    comments: article.commentCount || 0,
    views: article.views || 0,
    slug: article.slug,
    status: article.status,
    isLikedByCurrentUser: article.isLikedByCurrentUser || false
  };
};

// Helper function to get article content in the current language (keeping original interface)
export const getLocalizedArticleContent = (article, language = 'en') => {
  if (!article?.translations?.[language]) {
    // Fallback to English if translation not available
    console.log(`No ${language} translation found for article ${article?.id}, falling back to English`);
    return article?.translations?.['en'] || null;
  }
  console.log(`Using ${language} translation for article ${article?.id}`);
  return article.translations[language];
};

// Custom hook for articles
export const useArticles = () => {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Subscribe to global article updates
  useEffect(() => {
    console.log('useArticles: Subscribing to global cache updates');
    const unsubscribe = globalArticleCache.subscribe((articleId, updatedArticle) => {
      console.log('useArticles: Received update for article', articleId, updatedArticle);
      setArticles(currentArticles => {
        const updated = currentArticles.map(article => 
          article.id === articleId ? updatedArticle : article
        );
        console.log('useArticles: Updated articles count:', updated.length);
        return updated;
      });
    });
    
    // Cleanup subscription on unmount
    return () => {
      console.log('useArticles: Unsubscribing from global cache updates');
      unsubscribe();
    };
  }, []);

  const fetchAllArticles = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      console.log('Fetching all articles...');
      const response = await articlesAPI.getAll({ status: 'published' });
      console.log('All articles response:', response.data);
      const transformedArticles = response.data.articles?.map(transformArticle) || [];
      console.log('Transformed articles:', transformedArticles);
      
      // Cache articles globally
      globalArticleCache.cacheArticles(transformedArticles);
      
      setArticles(transformedArticles);
      return transformedArticles;
    } catch (err) {
      console.error('Error fetching all articles:', err);
      setError(err.response?.data?.message || 'Failed to fetch articles');
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchArticlesByCategory = useCallback(async (category) => {
    try {
      setLoading(true);
      setError(null);
      console.log('Fetching articles for category:', category);
      const response = await articlesAPI.getByType(category, { status: 'published' });
      console.log('Category articles response:', response.data);
      const transformedArticles = response.data.articles?.map(transformArticle) || [];
      console.log('Transformed category articles:', transformedArticles);
      
      // Cache articles globally
      globalArticleCache.cacheArticles(transformedArticles);
      
      return transformedArticles;
    } catch (err) {
      console.error('Error fetching articles by category:', err);
      setError(err.response?.data?.message || 'Failed to fetch articles');
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchArticleById = useCallback(async (id) => {
    try {
      setLoading(true);
      setError(null);
      console.log('Fetching article by ID/slug:', id);
      const response = await articlesAPI.getBySlug(id); // Try slug first
      console.log('Article response:', response.data);
      const transformedArticle = transformArticle(response.data);
      
      // Cache the article globally
      if (transformedArticle) {
        globalArticleCache.cacheArticles([transformedArticle]);
      }
      
      return transformedArticle;
    } catch (err) {
      // If slug fails, try ID
      try {
        console.log('Slug failed, trying ID:', id);
        const response = await articlesAPI.getById(id);
        console.log('Article by ID response:', response.data);
        const transformedArticle = transformArticle(response.data);
        
        // Cache the article globally
        if (transformedArticle) {
          globalArticleCache.cacheArticles([transformedArticle]);
        }
        
        return transformedArticle;
      } catch (secondErr) {
        console.error('Error fetching article by ID/slug:', err, secondErr);
        setError(secondErr.response?.data?.message || 'Failed to fetch article');
        return null;
      }
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    articles,
    loading,
    error,
    fetchAllArticles,
    fetchArticlesByCategory,
    fetchArticleById
  };
};

// Standalone utility functions (to maintain compatibility with existing code)
export const getAllArticles = async () => {
  try {
    console.log('Standalone getAllArticles called');
    const response = await articlesAPI.getAll({ status: 'published' });
    console.log('Standalone all articles response:', response.data);
    const transformedArticles = response.data.articles?.map(transformArticle) || [];
    
    // Cache articles globally
    globalArticleCache.cacheArticles(transformedArticles);
    
    return transformedArticles;
  } catch (err) {
    console.error('Error in standalone getAllArticles:', err);
    return [];
  }
};

export const getArticlesByCategory = async (category) => {
  try {
    console.log('Standalone getArticlesByCategory called for:', category);
    const response = await articlesAPI.getByType(category, { status: 'published' });
    console.log('Standalone category articles response:', response.data);
    const transformedArticles = response.data.articles?.map(transformArticle) || [];
    
    // Cache articles globally
    globalArticleCache.cacheArticles(transformedArticles);
    
    return transformedArticles;
  } catch (err) {
    console.error('Error in standalone getArticlesByCategory:', err);
    return [];
  }
};

export const getArticleById = async (id) => {
  try {
    console.log('Standalone getArticleById called for:', id);
    const response = await articlesAPI.getBySlug(id);
    console.log('Standalone article response:', response.data);
    const transformedArticle = transformArticle(response.data);
    
    // Cache the article globally
    if (transformedArticle) {
      globalArticleCache.cacheArticles([transformedArticle]);
    }
    
    return transformedArticle;
  } catch (err) {
    try {
      const response = await articlesAPI.getById(id);
      const transformedArticle = transformArticle(response.data);
      
      // Cache the article globally
      if (transformedArticle) {
        globalArticleCache.cacheArticles([transformedArticle]);
      }
      
      return transformedArticle;
    } catch (secondErr) {
      console.error('Error in standalone getArticleById:', err, secondErr);
      return null;
    }
  }
}; 