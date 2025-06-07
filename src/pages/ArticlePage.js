import React, { useState, useEffect } from 'react';
import { useParams, Navigate } from 'react-router-dom';
import Article from '../components/Article';
import { useArticles } from '../hooks/useArticles';

function ArticlePage() {
  const { id } = useParams();
  const { fetchArticleById, loading, error } = useArticles();
  const [article, setArticle] = useState(null);
  const [notFound, setNotFound] = useState(false);
  const [isInitialLoad, setIsInitialLoad] = useState(true);

  useEffect(() => {
    const loadArticle = async () => {
      try {
        console.log('ArticlePage: Loading article with ID/slug:', id);
        setIsInitialLoad(false);
        const articleData = await fetchArticleById(id);
        console.log('ArticlePage: Received article data:', articleData);
        
        if (articleData) {
          setArticle(articleData);
          setNotFound(false);
        } else {
          console.log('ArticlePage: No article data received');
          setNotFound(true);
        }
      } catch (err) {
        console.error('ArticlePage: Error loading article:', err);
        setNotFound(true);
      }
    };

    if (id) {
      loadArticle();
    }
  }, [id, fetchArticleById]);

  // Show loading spinner during initial load or while fetching
  if (loading || isInitialLoad) {
    return (
      <div className="py-8 flex items-center justify-center min-h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-gray-900 dark:border-gray-100 mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading article...</p>
        </div>
      </div>
    );
  }

  // Only redirect to 404 if we've finished loading and definitely have no article
  if (!loading && (error || notFound || !article)) {
    console.log('ArticlePage: Redirecting to 404 - error:', error, 'notFound:', notFound, 'article:', !!article);
    return <Navigate to="/404" replace />;
  }

  // Show article if we have it
  if (article) {
  return (
    <div className="py-8">
      <Article article={article} />
      </div>
    );
  }

  // Fallback loading state
  return (
    <div className="py-8 flex items-center justify-center min-h-96">
      <div className="text-center">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-gray-900 dark:border-gray-100 mx-auto"></div>
        <p className="mt-4 text-gray-600 dark:text-gray-400">Loading article...</p>
      </div>
    </div>
  );
}

export default ArticlePage; 