import React, { useState, useEffect } from 'react';
import { FiMail, FiTwitter, FiPenTool, FiCoffee, FiBriefcase, FiBookOpen, FiLinkedin, FiCheckSquare, FiFacebook, FiTarget, FiHeart, FiUsers, FiGlobe, FiBarChart2, FiAward, FiTrendingUp, FiCopy, FiCheck } from 'react-icons/fi';
import { useTranslation } from 'react-i18next';
import Newsletter from '../components/Newsletter';

function About() {
  const { t, i18n } = useTranslation();
  const [animatedStats, setAnimatedStats] = useState([0, 0, 0, 0]);
  const [isVisible, setIsVisible] = useState(false);
  const [emailCopied, setEmailCopied] = useState(false);
  const [showEmailTooltip, setShowEmailTooltip] = useState(false);
  
  const isRTL = i18n.language === 'ar';
  const emailAddress = 'sbhsportslab@gmail.com';

  // Stats data
  const stats = [
    { 
      key: 'viewers',
      value: 15, 
      suffix: 'M+', 
      label: t('Monthly Viewers on all platforms'),
      icon: FiBarChart2,
      color: 'text-blue-500'
    },
    { 
      key: 'articles',
      value: 1000, 
      suffix: '+', 
      label: t('Articles Published'),
      icon: FiBookOpen,
      color: 'text-blue-600'
    },
    { 
      key: 'experience',
      value: 8, 
      suffix: '', 
      label: t('Years Experience'),
      icon: FiBriefcase,
      color: 'text-blue-700'
    },
    { 
      key: 'languages',
      value: 3, 
      suffix: '', 
      label: t('Languages'),
      icon: FiGlobe,
      color: 'text-blue-800'
    }
  ];

  const values = [
    {
      icon: FiAward,
      title: t('Depth and accuracy over hype'),
      description: t('Depth Description')
    },
    {
      icon: FiCheckSquare,
      title: t('Authenticity & Integrity'),
      description: t('Authenticity & Integrity Description')
    },
    {
      icon: FiUsers,
      title: t('Community First and foremost'),
      description: t('Community Description')
    },
    {
      icon: FiCoffee,
      title: t('Passion for the game'),
      description: t('Multilingual Description')
    }
  ];

  // Animation effect for stats
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    const element = document.getElementById('stats-section');
    if (element) {
      observer.observe(element);
    }

    return () => {
      if (element) {
        observer.unobserve(element);
      }
    };
  }, []);

  useEffect(() => {
    if (isVisible) {
      stats.forEach((stat, index) => {
        let start = 0;
        const end = stat.value;
        const duration = 2000;
        const increment = end / (duration / 16);

        const timer = setInterval(() => {
          start += increment;
          if (start >= end) {
            start = end;
            clearInterval(timer);
          }
          setAnimatedStats(prev => {
            const newStats = [...prev];
            newStats[index] = Math.floor(start);
            return newStats;
          });
        }, 16);
      });
    }
  }, [isVisible]);

  const handleEmailCopy = async () => {
    try {
      await navigator.clipboard.writeText(emailAddress);
      setEmailCopied(true);
      setTimeout(() => setEmailCopied(false), 2000);
    } catch (err) {
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = emailAddress;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      setEmailCopied(true);
      setTimeout(() => setEmailCopied(false), 2000);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-slate-50 to-blue-100 dark:from-gray-900 dark:via-blue-900/20 dark:to-gray-900">
      <div className="container mx-auto px-4 py-12" dir={isRTL ? 'rtl' : 'ltr'}>
        
        {/* Hero Section */}
        <div className="text-center mb-16 relative">
          <div className="absolute inset-0 bg-blue-500/5 dark:bg-blue-400/10 rounded-full blur-3xl transform -translate-y-12"></div>
          <h1 className="text-6xl font-serif font-bold bg-gradient-to-r from-blue-600 via-blue-700 to-blue-800 dark:from-blue-400 dark:via-blue-300 dark:to-blue-200 bg-clip-text text-transparent mb-6 relative">
            {t('About')}
          </h1>
          <div className="w-24 h-1 bg-gradient-to-r from-blue-500 to-blue-700 mx-auto rounded-full"></div>
        </div>

        {/* Mission Section */}
        <section className="mb-20">
          <div className="max-w-6xl mx-auto">
            <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-lg rounded-3xl p-12 shadow-2xl border border-blue-100 dark:border-blue-800/30 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-blue-500/20 to-transparent rounded-full transform translate-x-16 -translate-y-16"></div>
              <div className="absolute bottom-0 left-0 w-24 h-24 bg-gradient-to-tr from-blue-600/10 to-transparent rounded-full transform -translate-x-12 translate-y-12"></div>
              
              <div className="relative">
                <div className="flex items-center gap-4 mb-8">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-700 rounded-2xl flex items-center justify-center shadow-lg">
                    <FiPenTool className="w-8 h-8 text-white" />
                  </div>
                  <h2 className="text-4xl font-serif font-bold text-blue-800 dark:text-blue-200">
                    {t('Our Mission')}
                  </h2>
                </div>
                
                <div className="text-lg text-gray-700 dark:text-gray-300 leading-relaxed font-medium">
                  {t('Mission Description').split('\n\n').map((paragraph, index) => (
                    <p key={index} className="mb-4 last:mb-0">
                      {paragraph}
                    </p>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Values Section */}
        <section className="mb-20">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-serif font-bold text-blue-800 dark:text-blue-200 mb-4">
              {t('Our Values')}
            </h2>
            <div className="w-16 h-1 bg-gradient-to-r from-blue-500 to-blue-700 mx-auto rounded-full"></div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 max-w-7xl mx-auto">
            {values.map((value, index) => (
              <div 
                key={index}
                className="group relative bg-white/70 dark:bg-gray-800/70 backdrop-blur-lg rounded-2xl p-8 shadow-lg border border-blue-100 dark:border-blue-800/30 hover:shadow-2xl hover:scale-105 transition-all duration-500 hover:bg-white dark:hover:bg-gray-800"
                style={{ animationDelay: `${index * 150}ms` }}
              >
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-blue-700/5 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                
                <div className="relative">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-700 rounded-xl flex items-center justify-center mb-6 shadow-lg group-hover:scale-110 transition-transform duration-300">
                    <value.icon className="w-8 h-8 text-white" />
                  </div>
                  
                  <h3 className="text-xl font-bold text-blue-800 dark:text-blue-200 mb-3 group-hover:text-blue-600 dark:group-hover:text-blue-300 transition-colors">
                    {value.title}
                  </h3>
                  
                  <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
                    {value.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Stats Section */}
        <section id="stats-section" className="mb-20">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-serif font-bold text-blue-800 dark:text-blue-200 mb-4">
              {t('By the Numbers')}
            </h2>
            <div className="w-16 h-1 bg-gradient-to-r from-blue-500 to-blue-700 mx-auto rounded-full"></div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 max-w-6xl mx-auto">
            {stats.map((stat, index) => (
              <div 
                key={stat.key}
                className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-lg rounded-2xl p-8 shadow-xl border border-blue-100 dark:border-blue-800/30 text-center group hover:scale-105 transition-all duration-500 relative overflow-hidden"
              >
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-blue-700/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                
                <div className="relative">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-700 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg group-hover:shadow-2xl transition-all duration-300">
                    <stat.icon className="w-8 h-8 text-white" />
                  </div>
                  
                  <div className="text-4xl font-bold text-blue-700 dark:text-blue-300 mb-2">
                    {animatedStats[index]}{stat.suffix}
                  </div>
                  
                  <div className="text-gray-600 dark:text-gray-400 font-medium">
                    {stat.label}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Team Section */}
        <section className="mb-20">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-serif font-bold text-blue-800 dark:text-blue-200 mb-4">
              {t('The Team')}
            </h2>
            <div className="w-16 h-1 bg-gradient-to-r from-blue-500 to-blue-700 mx-auto rounded-full"></div>
          </div>
          
          <div className="max-w-4xl mx-auto">
            <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-lg rounded-3xl p-12 shadow-2xl border border-blue-100 dark:border-blue-800/30 relative overflow-hidden">
              <div className="absolute top-0 left-0 w-40 h-40 bg-gradient-to-br from-blue-500/10 to-transparent rounded-full transform -translate-x-20 -translate-y-20"></div>
              <div className="absolute bottom-0 right-0 w-32 h-32 bg-gradient-to-tl from-blue-600/10 to-transparent rounded-full transform translate-x-16 translate-y-16"></div>
              
              <div className="relative flex flex-col md:flex-row items-center gap-12" dir={isRTL ? 'rtl' : 'ltr'}>
                <div className="flex-shrink-0">
                  <div className="w-32 h-32 bg-gradient-to-br from-blue-500 to-blue-700 rounded-3xl flex items-center justify-center shadow-2xl">
                    <span className="text-4xl font-bold text-white">SBH</span>
                  </div>
                </div>
                
                <div className={`flex-1 text-center ${isRTL ? 'md:text-right' : 'md:text-left'}`}>
                  <h3 className="text-3xl font-serif font-bold text-blue-800 dark:text-blue-200 mb-2">
                    {i18n.language === 'ar' ? 'صدقي بن حوالة' : 'Sedki Ben Haouala'}
                  </h3>
                  
                  <p className="text-blue-600 dark:text-blue-400 font-semibold mb-6">
                    {t('Founder Title')}
                  </p>
                  
                  <p className="text-gray-700 dark:text-gray-300 leading-relaxed text-lg">
                    {t('Founder Bio')}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Contact Section */}
        <section className="mb-16">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-serif font-bold text-blue-800 dark:text-blue-200 mb-4">
              {t('Get in Touch')}
            </h2>
            <div className="w-16 h-1 bg-gradient-to-r from-blue-500 to-blue-700 mx-auto rounded-full"></div>
          </div>
          
          <div className="max-w-4xl mx-auto">
            <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-lg rounded-3xl p-12 shadow-2xl border border-blue-100 dark:border-blue-800/30 relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-blue-700/5 rounded-3xl"></div>
              
              <div className="relative text-center">
                <p className="text-lg text-gray-700 dark:text-gray-300 mb-8 leading-relaxed">
                  {t('Contact Description')}
                </p>
                
                <div className="flex justify-center gap-6 flex-wrap">
                  <div className="relative">
                    <button 
                      onClick={handleEmailCopy}
                      onMouseEnter={() => setShowEmailTooltip(true)}
                      onMouseLeave={() => setShowEmailTooltip(false)}
                      className="group flex items-center gap-3 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white px-6 py-3 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105"
                    >
                      {emailCopied ? (
                        <FiCheck className="w-5 h-5 group-hover:scale-110 transition-transform" />
                      ) : (
                        <FiMail className="w-5 h-5 group-hover:scale-110 transition-transform" />
                      )}
                      {emailCopied ? 'Copied!' : t('Email')}
                    </button>
                    
                    {/* Email tooltip */}
                    {showEmailTooltip && !emailCopied && (
                      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 dark:bg-gray-700 text-white text-sm rounded-lg shadow-lg z-10">
                        <div className="flex items-center gap-2">
                          <span>{emailAddress}</span>
                          <FiCopy className="w-4 h-4" />
                        </div>
                        <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900 dark:border-t-gray-700"></div>
                      </div>
                    )}
                  </div>
                  
                  <a 
                    href="https://x.com/sedki25"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group flex items-center gap-3 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white px-6 py-3 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105"
                  >
                    <FiTwitter className="w-5 h-5 group-hover:scale-110 transition-transform" />
                    {t('Twitter')}
                  </a>
                  
                  <a 
                    href="https://www.facebook.com/profile.php?id=100009033914272"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group flex items-center gap-3 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white px-6 py-3 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105"
                  >
                    <FiFacebook className="w-5 h-5 group-hover:scale-110 transition-transform" />
                    {t('Facebook')}
                  </a>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Newsletter Section */}
        <div className="max-w-4xl mx-auto">
          <Newsletter variant="about" />
        </div>
      </div>
    </div>
  );
}

export default About; 