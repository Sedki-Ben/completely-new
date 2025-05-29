import { Link } from 'react-router-dom';
import { FiArrowRight, FiHeart, FiMessageCircle } from 'react-icons/fi';
import { useTranslation } from 'react-i18next';
import Newsletter from '../components/Newsletter';

import heroImage from '../assets/images/heroes/football-stadium.jpg';
import article1Image from '../assets/images/articles/champions-league.jpg';
import article2Image from '../assets/images/articles/world-cup.jpg';
import article3Image from '../assets/images/articles/premier-league.png';

function Home() {
  const { t } = useTranslation();
  const popularArticles = [
    {
      id: 1,
      title: "Tactical Analysis: Champions League Final",
      author: "John Doe",
      date: "January 15, 2024",
      image: article1Image,
      excerpt: "Breaking down the key moments that decided Europe's biggest game",
      category: "analysis",
      likes: 245,
      comments: 58
    },
    {
      id: 2,
      title: "World Cup 2026: Early Predictions",
      author: "Jane Smith",
      date: "January 10, 2024",
      image: article2Image,
      excerpt: "Which teams are shaping up to be contenders in North America",
      category: "stories",
      likes: 189,
      comments: 42
    },
    {
      id: 3,
      title: "Premier League Title Race Heats Up",
      author: "Mike Johnson",
      date: "January 5, 2024",
      image: article3Image,
      excerpt: "Why this might be the closest title race in recent memory",
      category: "analysis",
      likes: 312,
      comments: 76
    }
  ];

  return (
    <div className="container mx-auto px-4">
      {/* Hero Section with optimized image */}
      <section className="relative h-[60vh] md:h-[80vh] bg-gray-900 rounded-xl overflow-hidden mb-16">
        <div className="relative w-full h-full">
          <img 
            src={heroImage} 
            alt="Football stadium at night" 
            className="absolute inset-0 w-full h-full object-cover object-center transform hover:scale-105 transition-transform duration-700"
            loading="eager"
            sizes="(max-width: 768px) 100vw, (max-width: 1200px) 80vw, 1200px"
          />
          <div className="absolute inset-0 bg-gradient-to-r from-black/80 to-black/40"></div>
          <div className="relative h-full flex flex-col justify-center px-6 lg:px-12">
            <div className="max-w-3xl bg-white/95 p-8 rounded-lg shadow-xl transform hover:-translate-y-1 transition-transform duration-300">
              <h2 className="text-3xl md:text-4xl lg:text-5xl font-serif text-gray-900 mb-4 leading-tight">
                The Beautiful Game Through My Lens
              </h2>
              <p className="text-lg md:text-xl text-gray-700 mb-8 leading-relaxed">
                Analysis, stories, and insights from the world of football
              </p>
              <Link 
                to="/analysis" 
                className="inline-flex items-center space-x-2 px-6 py-3 bg-blue-900 text-white rounded-md hover:bg-blue-800 transform hover:-translate-y-0.5 transition-all duration-300"
              >
                <span>Read the Latest</span>
                <FiArrowRight className="ml-2" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Most Popular Section with enhanced article cards */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto">
          <h3 className="text-3xl font-serif mb-12 inline-block relative after:content-[''] after:absolute after:bottom-0 after:left-0 after:w-1/2 after:h-1 after:bg-blue-900">
            Most Popular
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {popularArticles.map((article) => (
              <Link 
                to={`/article/${article.id}`} 
                key={article.id} 
                className="group bg-white rounded-xl overflow-hidden shadow-lg hover:shadow-xl transform hover:-translate-y-1 transition-all duration-300"
              >
                <div className="aspect-w-16 aspect-h-9 overflow-hidden">
                  <img 
                    src={article.image} 
                    alt={article.title}
                    className="w-full h-full object-cover object-center transform group-hover:scale-110 transition-transform duration-700"
                    loading="lazy"
                    sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                  />
                </div>
                <div className="p-6">
                  <div className="text-sm text-blue-900 font-semibold mb-2 uppercase tracking-wider">
                    {article.category}
                  </div>
                  <h4 className="font-serif text-xl mb-3 group-hover:text-blue-900 transition-colors duration-300">
                    {article.title}
                  </h4>
                  <p className="text-sm text-gray-600 mb-4">
                    {article.author} • {article.date}
                  </p>
                  <p className="text-gray-700">{article.excerpt}</p>
                  <div className="mt-4 flex items-center justify-between">
                    <div className="flex items-center space-x-4 text-gray-600">
                      <span className="flex items-center">
                        <FiHeart className="mr-1" /> {article.likes}
                      </span>
                      <span className="flex items-center">
                        <FiMessageCircle className="mr-1" /> {article.comments}
                      </span>
                    </div>
                    <span className="text-blue-900 font-medium group-hover:translate-x-2 transition-transform duration-300">
                      Read more →
                    </span>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Newsletter Section */}
      <section className="py-16">
        <div className="max-w-4xl mx-auto">
          <Newsletter />
        </div>
      </section>
    </div>
  );
}

export default Home;