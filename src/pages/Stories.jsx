import React from 'react';
import { Link } from 'react-router-dom';

function Stories() {
  const stories = [
    {
      title: "From Sunday League to Premier League",
      author: "James Wilson",
      date: "March 12, 2024",
      category: "Player Journey",
      excerpt: "The incredible journey of Marcus Thompson, who went from playing amateur football to signing with a Premier League club at age 23...",
      readTime: "8 min read"
    },
    {
      title: "The Miracle of Istanbul",
      author: "Sarah Chen",
      date: "March 8, 2024",
      category: "Historic Matches",
      excerpt: "Revisiting the legendary 2005 Champions League final between Liverpool and AC Milan, where the impossible became possible...",
      readTime: "12 min read"
    },
    {
      title: "Football in the Amazon",
      author: "Carlos Rodriguez",
      date: "March 1, 2024",
      category: "Global Football",
      excerpt: "Discovering how indigenous communities in the Amazon rainforest have embraced football and created their own unique style...",
      readTime: "10 min read"
    }
  ];

  return (
    <div className="max-w-4xl mx-auto py-8">
      <h1 className="text-3xl font-bold mb-8">Football Stories</h1>
      
      <div className="space-y-8">
        {stories.map((story, index) => (
          <article key={index} className="bg-white p-6 rounded-lg shadow-md">
            <div className="flex justify-between items-start mb-4">
              <div>
                <span className="bg-blue-100 text-blue-900 px-3 py-1 rounded-full text-sm">
                  {story.category}
                </span>
                <h2 className="text-2xl font-semibold mt-2">{story.title}</h2>
              </div>
              <span className="text-gray-500 text-sm">{story.readTime}</span>
            </div>
            
            <p className="text-gray-800 mb-4">{story.excerpt}</p>
            
            <div className="flex justify-between items-center">
              <div className="text-sm text-gray-600">
                By {story.author} • {story.date}
              </div>
              <button className="text-blue-900 hover:underline">
                Read More →
              </button>
            </div>
          </article>
        ))}
      </div>

      <div className="mt-8 text-center">
        <Link to="/" className="text-blue-900 hover:underline">
          Back to Home
        </Link>
      </div>
    </div>
  );
}

export default Stories;

