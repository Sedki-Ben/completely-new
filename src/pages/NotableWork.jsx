import React from 'react';
import { Link } from 'react-router-dom';

function NotableWork() {
  const works = [
    {
      title: "The Evolution of Modern Football Tactics",
      type: "Research Paper",
      year: "2024",
      description: "A comprehensive study of how football tactics have evolved over the past decade, featuring interviews with top managers and statistical analysis.",
      impact: "Featured in UEFA Technical Report"
    },
    {
      title: "Data Analytics in Football Scouting",
      type: "Case Study",
      year: "2023",
      description: "How data analytics is revolutionizing player scouting and recruitment in professional football.",
      impact: "Implemented by 3 Premier League clubs"
    },
    {
      title: "Youth Development in European Football",
      type: "Documentary Series",
      year: "2023",
      description: "A six-part series exploring the most successful youth academies in European football.",
      impact: "Over 1 million views"
    }
  ];

  return (
    <div className="max-w-4xl mx-auto py-8">
      <h1 className="text-3xl font-bold mb-8">Notable Work</h1>
      
      <div className="grid gap-8">
        {works.map((work, index) => (
          <div key={index} className="bg-white p-6 rounded-lg shadow-md border-l-4 border-blue-900">
            <div className="flex justify-between items-start mb-4">
              <h2 className="text-2xl font-semibold">{work.title}</h2>
              <div className="flex items-center space-x-2">
                <span className="bg-gray-100 text-gray-800 px-3 py-1 rounded-full text-sm">
                  {work.type}
                </span>
                <span className="bg-blue-100 text-blue-900 px-3 py-1 rounded-full text-sm">
                  {work.year}
                </span>
              </div>
            </div>
            
            <p className="text-gray-800 mb-4">{work.description}</p>
            
            <div className="flex items-center justify-between">
              <div className="text-green-700 font-medium">
                Impact: {work.impact}
              </div>
              <button className="bg-blue-900 text-white px-4 py-2 rounded hover:bg-blue-800">
                View Details
              </button>
            </div>
          </div>
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

export default NotableWork;

