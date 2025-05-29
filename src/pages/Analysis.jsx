import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { FiHeart, FiMessageCircle, FiShare2 } from 'react-icons/fi';

function Analysis() {
  const [analyses, setAnalyses] = useState([
    {
      id: 1,
      title: "Modern Pressing Tactics",
      date: "March 15, 2024",
      content: "In modern football, pressing has become a crucial aspect of team tactics. Teams like Manchester City and Liverpool have revolutionized how pressure is applied in the final third...",
      likes: 156,
      comments: 23,
      isLiked: false,
      commentsList: [
        { id: 1, user: "TacticalMind", text: "Great analysis! The evolution of gegenpressing has been fascinating.", time: "2 hours ago" },
        { id: 2, user: "FootballPro", text: "Would love to see more examples from different leagues.", time: "5 hours ago" }
      ]
    },
    {
      id: 2,
      title: "The Evolution of the False 9",
      date: "March 10, 2024",
      content: "From Messi to Firmino, the false 9 position has transformed how teams approach attacking play. This deep-lying forward role creates unique tactical advantages...",
      likes: 203,
      comments: 31,
      isLiked: false,
      commentsList: [
        { id: 1, user: "MessiFan", text: "Messi really pioneered this role under Guardiola.", time: "1 day ago" }
      ]
    },
    {
      id: 3,
      title: "Wing-Back Revolution",
      date: "March 5, 2024",
      content: "The modern wing-back must balance defensive duties with attacking prowess. This analysis explores how teams utilize wing-backs in both phases of play...",
      likes: 178,
      comments: 27,
      isLiked: false,
      commentsList: []
    }
  ]);

  const [newComments, setNewComments] = useState({});
  const [showComments, setShowComments] = useState({});

  const handleLike = (analysisId) => {
    setAnalyses(analyses.map(analysis => {
      if (analysis.id === analysisId) {
        return {
          ...analysis,
          likes: analysis.isLiked ? analysis.likes - 1 : analysis.likes + 1,
          isLiked: !analysis.isLiked
        };
      }
      return analysis;
    }));
  };

  const handleComment = (analysisId) => {
    if (!newComments[analysisId]) return;

    setAnalyses(analyses.map(analysis => {
      if (analysis.id === analysisId) {
        return {
          ...analysis,
          comments: analysis.comments + 1,
          commentsList: [
            {
              id: analysis.commentsList.length + 1,
              user: "CurrentUser",
              text: newComments[analysisId],
              time: "Just now"
            },
            ...analysis.commentsList
          ]
        };
      }
      return analysis;
    }));

    setNewComments({ ...newComments, [analysisId]: "" });
  };

  const toggleComments = (analysisId) => {
    setShowComments({ ...showComments, [analysisId]: !showComments[analysisId] });
  };

  return (
    <div className="max-w-4xl mx-auto py-8">
      <h1 className="text-3xl font-bold mb-8">Tactical Analysis</h1>
      
      <div className="grid gap-8">
        {analyses.map((analysis) => (
          <div key={analysis.id} className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-2">{analysis.title}</h2>
            <p className="text-gray-600 text-sm mb-4">{analysis.date}</p>
            <p className="text-gray-800 mb-6">{analysis.content}</p>
            
            <div className="border-t pt-4">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-6">
                  <button 
                    onClick={() => handleLike(analysis.id)}
                    className={`flex items-center space-x-2 ${analysis.isLiked ? 'text-blue-500' : 'text-gray-600'} hover:text-blue-500 transition-colors`}
                  >
                    <FiHeart className={`w-5 h-5 ${analysis.isLiked ? 'fill-current' : ''}`} />
                    <span>{analysis.likes}</span>
                  </button>
                  <button 
                    onClick={() => toggleComments(analysis.id)}
                    className="flex items-center space-x-2 text-gray-600 hover:text-blue-900 transition-colors"
                  >
                    <FiMessageCircle className="w-5 h-5" />
                    <span>{analysis.comments}</span>
                  </button>
                  <button className="flex items-center space-x-2 text-gray-600 hover:text-blue-900 transition-colors">
                    <FiShare2 className="w-5 h-5" />
                    <span>Share</span>
                  </button>
                </div>
                <button className="bg-blue-900 text-white px-4 py-2 rounded hover:bg-blue-800">
                  Read Full Analysis
                </button>
              </div>

              {showComments[analysis.id] && (
                <div className="mt-4 space-y-4">
                  <div className="flex items-start space-x-2">
                    <input
                      type="text"
                      value={newComments[analysis.id] || ""}
                      onChange={(e) => setNewComments({ ...newComments, [analysis.id]: e.target.value })}
                      placeholder="Add a comment..."
                      className="flex-1 p-2 border rounded-lg"
                    />
                    <button
                      onClick={() => handleComment(analysis.id)}
                      className="px-4 py-2 bg-blue-900 text-white rounded-lg hover:bg-blue-800"
                    >
                      Post
                    </button>
                  </div>
                  
                  <div className="space-y-3">
                    {analysis.commentsList.map((comment) => (
                      <div key={comment.id} className="bg-gray-50 p-3 rounded-lg">
                        <div className="flex justify-between items-center mb-1">
                          <span className="font-medium text-blue-900">{comment.user}</span>
                          <span className="text-sm text-gray-500">{comment.time}</span>
                        </div>
                        <p className="text-gray-700">{comment.text}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
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

export default Analysis;
