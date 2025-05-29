import React from 'react';
import { useParams, Navigate } from 'react-router-dom';
import Article from '../components/Article';
import { getArticleById } from '../data/articles';

function ArticlePage() {
  const { id } = useParams();
  const article = getArticleById(id);

  if (!article) {
    return <Navigate to="/404" replace />;
  }

  return (
    <div className="py-8">
      <Article article={article} />
    </div>
  );
}

export default ArticlePage; 