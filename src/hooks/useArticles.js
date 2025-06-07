import { useState, useCallback } from 'react';
import { articles as articlesAPI } from '../services/api';

// Default author info to maintain consistency
const DEFAULT_AUTHOR = "Sedki B.Haouala";
const DEFAULT_AUTHOR_IMAGE = "/uploads/profile/bild3.jpg"; // Backend path

// Category translations (keeping from original structure)
export const categoryTranslations = {
  'etoile-du-sahel': { 
    en: 'Etoile Du Sahel', 
    fr: 'Étoile du Sahel', 
    ar: 'النجم الساحلي' 
  },
  'the-beautiful-game': { 
    en: 'The Beautiful Game', 
    fr: 'Le Beau Jeu', 
    ar: 'اللعبة الجميلة' 
  },
  'all-sports-hub': { 
    en: 'All-Sports Hub', 
    fr: 'Hub Tous Sports', 
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
  
  return {
    id: article._id,
    translations: article.translations,
    author: article.author?.name || DEFAULT_AUTHOR,
    authorImage: article.authorImage ? `${backendUrl}${article.authorImage}` : `${backendUrl}${DEFAULT_AUTHOR_IMAGE}`,
    date: new Date(article.publishedAt || article.createdAt).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    }),
    image: article.image ? `${backendUrl}${article.image}` : null,
    category: article.category,
    likes: article.likes?.count || 0,
    comments: article.commentCount || 0,
    views: article.views || 0,
    slug: article.slug,
    status: article.status
  };
};

// Helper function to get article content in the current language (keeping original interface)
export const getLocalizedArticleContent = (article, language = 'en') => {
  if (!article?.translations?.[language]) {
    // Fallback to English if translation not available
    return article?.translations?.['en'] || null;
  }
  return article.translations[language];
};

// Custom hook for articles
export const useArticles = () => {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchAllArticles = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      console.log('Fetching all articles...');
      const response = await articlesAPI.getAll({ status: 'published' });
      console.log('All articles response:', response.data);
      const transformedArticles = response.data.articles?.map(transformArticle) || [];
      console.log('Transformed articles:', transformedArticles);
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
      return transformArticle(response.data);
    } catch (err) {
      // If slug fails, try ID
      try {
        console.log('Slug failed, trying ID:', id);
        const response = await articlesAPI.getById(id);
        console.log('Article by ID response:', response.data);
        return transformArticle(response.data);
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
    return response.data.articles?.map(transformArticle) || [];
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
    return response.data.articles?.map(transformArticle) || [];
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
    return transformArticle(response.data);
  } catch (err) {
    try {
      const response = await articlesAPI.getById(id);
      return transformArticle(response.data);
    } catch (secondErr) {
      console.error('Error in standalone getArticleById:', err, secondErr);
      return null;
    }
  }
}; 