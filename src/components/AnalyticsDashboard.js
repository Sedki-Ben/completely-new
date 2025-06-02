import React, { useState, useEffect } from 'react';
import { analytics } from '../services/analytics';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
} from 'chart.js';
import { Line, Bar, Pie } from 'react-chartjs-2';
import { FiTrendingUp, FiEye, FiHeart, FiShare2, FiMessageSquare, FiDownload, FiCalendar } from 'react-icons/fi';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const AnalyticsDashboard = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeRange, setTimeRange] = useState('week'); // 'week', 'month', 'year'

  useEffect(() => {
    fetchAnalytics();
  }, [timeRange]);

  const fetchAnalytics = async () => {
    try {
      setLoading(true);
      const analyticsData = await analytics.getWriterAnalytics();
      setData(analyticsData);
    } catch (err) {
      setError('Failed to load analytics data');
      console.error('Analytics error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleExport = () => {
    if (!data) return;
    
    const csvContent = "data:text/csv;charset=utf-8,"
      + "Article,Views,Likes,Shares,Comments\n"
      + data.map(article => {
          const views = article.metrics.find(m => m.type === 'view')?.count || 0;
          const likes = article.metrics.find(m => m.type === 'interaction_like')?.count || 0;
          const shares = article.metrics.find(m => m.type === 'interaction_share')?.count || 0;
          const comments = article.metrics.find(m => m.type === 'interaction_comment')?.count || 0;
          return `Article ${article._id},${views},${likes},${shares},${comments}`;
      }).join("\n");
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "analytics.csv");
    document.body.appendChild(link);
    link.click();
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-100 dark:bg-red-900/50 text-red-600 dark:text-red-300 p-4 rounded-lg text-center">
        {error}
      </div>
    );
  }

  // Prepare data for charts
  const prepareChartData = () => {
    if (!data) return null;

    const viewsData = [];
    const likesData = [];
    const sharesData = [];
    const commentsData = [];

    data.forEach(article => {
      article.metrics.forEach(metric => {
        switch (metric.type) {
          case 'view':
            viewsData.push(metric.count);
            break;
          case 'interaction_like':
            likesData.push(metric.count);
            break;
          case 'interaction_share':
            sharesData.push(metric.count);
            break;
          case 'interaction_comment':
            commentsData.push(metric.count);
            break;
          default:
            break;
        }
      });
    });

    return {
      views: viewsData.reduce((a, b) => a + b, 0),
      likes: likesData.reduce((a, b) => a + b, 0),
      shares: sharesData.reduce((a, b) => a + b, 0),
      comments: commentsData.reduce((a, b) => a + b, 0)
    };
  };

  const stats = prepareChartData();

  return (
    <div className="space-y-6 bg-white/80 dark:bg-slate-900/80 backdrop-blur-lg rounded-2xl shadow-2xl p-8 border border-blue-200 dark:border-blue-900">
      {/* Header Controls */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <FiCalendar className="text-blue-500 w-5 h-5" />
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="bg-white/50 dark:bg-slate-800/50 border border-blue-200 dark:border-blue-800 rounded-lg px-4 py-2 text-gray-700 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="week">Last 7 Days</option>
            <option value="month">Last 30 Days</option>
            <option value="year">Last Year</option>
          </select>
        </div>
        <button
          onClick={handleExport}
          className="flex items-center px-4 py-2 bg-emerald-500 hover:bg-emerald-600 text-white rounded-lg transition-colors"
        >
          <FiDownload className="w-5 h-5 mr-2" />
          Export Data
        </button>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Views"
          value={stats?.views || 0}
          icon={<FiEye className="w-6 h-6" />}
          color="blue"
          trend={15}
          timeRange={timeRange}
        />
        <StatCard
          title="Total Likes"
          value={stats?.likes || 0}
          icon={<FiHeart className="w-6 h-6" />}
          color="red"
          trend={8}
          timeRange={timeRange}
        />
        <StatCard
          title="Total Shares"
          value={stats?.shares || 0}
          icon={<FiShare2 className="w-6 h-6" />}
          color="green"
          trend={12}
          timeRange={timeRange}
        />
        <StatCard
          title="Total Comments"
          value={stats?.comments || 0}
          icon={<FiMessageSquare className="w-6 h-6" />}
          color="purple"
          trend={-3}
          timeRange={timeRange}
        />
      </div>

      {/* Engagement Chart */}
      <div className="bg-white/50 dark:bg-slate-800/50 rounded-xl shadow-lg p-6 border border-blue-100 dark:border-blue-900">
        <h3 className="text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200">Engagement Overview</h3>
        <Bar
          data={{
            labels: ['Views', 'Likes', 'Shares', 'Comments'],
            datasets: [
              {
                label: 'Count',
                data: [stats?.views || 0, stats?.likes || 0, stats?.shares || 0, stats?.comments || 0],
                backgroundColor: [
                  'rgba(59, 130, 246, 0.7)', // blue
                  'rgba(239, 68, 68, 0.7)',   // red
                  'rgba(34, 197, 94, 0.7)',   // green
                  'rgba(168, 85, 247, 0.7)',  // purple
                ],
                borderColor: [
                  'rgb(59, 130, 246)',
                  'rgb(239, 68, 68)',
                  'rgb(34, 197, 94)',
                  'rgb(168, 85, 247)',
                ],
                borderWidth: 2,
              },
            ],
          }}
          options={{
            responsive: true,
            plugins: {
              legend: {
                display: false,
              },
              tooltip: {
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                padding: 12,
                titleFont: {
                  size: 14,
                  weight: 'bold'
                },
                bodyFont: {
                  size: 13
                }
              }
            },
            scales: {
              y: {
                beginAtZero: true,
                grid: {
                  color: 'rgba(156, 163, 175, 0.1)',
                },
                ticks: {
                  color: 'rgb(156, 163, 175)',
                }
              },
              x: {
                grid: {
                  display: false
                },
                ticks: {
                  color: 'rgb(156, 163, 175)',
                }
              }
            },
          }}
        />
      </div>

      {/* Article Performance Table */}
      <div className="bg-white/50 dark:bg-slate-800/50 rounded-xl shadow-lg p-6 border border-blue-100 dark:border-blue-900">
        <h3 className="text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200">Article Performance</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-blue-200 dark:border-blue-800">
                <th className="text-left py-3 px-4 text-gray-600 dark:text-gray-400 font-semibold">Article</th>
                <th className="text-right py-3 px-4 text-gray-600 dark:text-gray-400 font-semibold">Views</th>
                <th className="text-right py-3 px-4 text-gray-600 dark:text-gray-400 font-semibold">Likes</th>
                <th className="text-right py-3 px-4 text-gray-600 dark:text-gray-400 font-semibold">Shares</th>
                <th className="text-right py-3 px-4 text-gray-600 dark:text-gray-400 font-semibold">Comments</th>
              </tr>
            </thead>
            <tbody>
              {data?.map((article, index) => (
                <tr key={index} className="border-b border-blue-100 dark:border-blue-900 hover:bg-blue-50 dark:hover:bg-blue-900/30 transition-colors">
                  <td className="py-3 px-4 text-gray-800 dark:text-gray-200">Article {index + 1}</td>
                  <td className="text-right py-3 px-4 text-gray-800 dark:text-gray-200">
                    {article.metrics.find(m => m.type === 'view')?.count.toLocaleString() || 0}
                  </td>
                  <td className="text-right py-3 px-4 text-gray-800 dark:text-gray-200">
                    {article.metrics.find(m => m.type === 'interaction_like')?.count.toLocaleString() || 0}
                  </td>
                  <td className="text-right py-3 px-4 text-gray-800 dark:text-gray-200">
                    {article.metrics.find(m => m.type === 'interaction_share')?.count.toLocaleString() || 0}
                  </td>
                  <td className="text-right py-3 px-4 text-gray-800 dark:text-gray-200">
                    {article.metrics.find(m => m.type === 'interaction_comment')?.count.toLocaleString() || 0}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

// Stat Card Component
const StatCard = ({ title, value, icon, color, trend, timeRange }) => {
  const colorClasses = {
    blue: 'bg-gradient-to-br from-blue-400 to-blue-600',
    red: 'bg-gradient-to-br from-red-400 to-red-600',
    green: 'bg-gradient-to-br from-emerald-400 to-emerald-600',
    purple: 'bg-gradient-to-br from-purple-400 to-purple-600',
  };

  return (
    <div className="bg-white/50 dark:bg-slate-800/50 rounded-xl shadow-lg p-6 border border-blue-100 dark:border-blue-900 hover:shadow-xl transition-shadow">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-600 dark:text-gray-400 font-medium">{title}</p>
          <p className="text-2xl font-bold mt-1 text-gray-800 dark:text-gray-200">{value.toLocaleString()}</p>
        </div>
        <div className={`${colorClasses[color]} p-3 rounded-lg text-white shadow-lg`}>
          {icon}
        </div>
      </div>
      <div className="mt-4">
        <div className="flex items-center text-sm">
          <FiTrendingUp className={trend >= 0 ? "text-emerald-500" : "text-red-500"} />
          <span className={trend >= 0 ? "text-emerald-500 ml-1" : "text-red-500 ml-1"}>
            {trend}%
          </span>
          <span className="text-gray-600 dark:text-gray-400 ml-2">vs last {timeRange}</span>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsDashboard; 