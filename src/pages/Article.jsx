import { useParams } from 'react-router-dom';
import { articles } from '../data/articles'; // You'll need to create this data file

function Article() {
  const { id } = useParams();
  const article = articles.find(article => article.id === parseInt(id));

  if (!article) {
    return <div>Article not found</div>;
  }

  return (
    <article className="max-w-4xl mx-auto py-16 px-6">
      <header className="mb-12">
        <h1 className="text-4xl md:text-5xl font-serif mb-4">{article.title}</h1>
        <div className="flex items-center space-x-4">
          <img 
            src={article.authorImage} 
            alt={article.author} 
            className="w-12 h-12 rounded-full"
          />
          <div>
            <p className="font-medium">{article.author}</p>
            <p className="text-gray-600 text-sm">{article.date} • {article.readTime} min read</p>
          </div>
        </div>
      </header>
      
      <img 
        src={article.image} 
        alt={article.title} 
        className="w-full h-auto mb-12 rounded-lg"
      />
      
      <div className="prose max-w-none">
        {article.content}
      </div>
      
      <footer className="mt-16 pt-8 border-t border-gray-200">
        <h3 className="text-xl font-serif mb-4">More from {article.author}</h3>
        {/* You could add related articles here */}
      </footer>
    </article>
  );
}

export default Article;