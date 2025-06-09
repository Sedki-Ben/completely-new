import React, { useState, useEffect } from 'react';
import { FiHeart, FiMessageCircle, FiBookmark, FiEdit3, FiTrash2 } from 'react-icons/fi';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { articles as articleApi } from '../services/api';
import { getLocalizedArticleContent, categoryTranslations, updateArticleCommentCount, useArticles } from '../hooks/useArticles';
import Newsletter from './Newsletter';
import CommentsSection from './CommentsSection';
import ArticleNavigation from './ArticleNavigation';

function Article({ article }) {
  const { i18n, t } = useTranslation();
  const { user, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const { getNavigationArticles } = useArticles();
  const [articleData, setArticleData] = useState(article);
  const [likeLoading, setLikeLoading] = useState(false);
  const [isLiked, setIsLiked] = useState(false);
  const [navigationArticles, setNavigationArticles] = useState({ previousArticle: null, nextArticle: null });
  const [navigationLoading, setNavigationLoading] = useState(false);
  
  const isAdmin = user?.role === 'admin';

  // Update article data when prop changes
  useEffect(() => {
    console.log('Article prop changed, updating article data:', article.id || article._id);
    setArticleData(article);
    // Reset navigation when article changes
    setNavigationArticles({ previousArticle: null, nextArticle: null });
    // Only set loading if we actually have a category (avoid unnecessary loading states)
    if (article && article.category) {
      setNavigationLoading(true);
    } else {
      setNavigationLoading(false);
    }
  }, [article]);

  // Check if user has liked this article when user or article changes
  useEffect(() => {
    console.log('Like state initialization:', {
      user: user?._id,
      isLikedByCurrentUser: articleData.isLikedByCurrentUser,
      articleId: articleData._id
    });
    
    if (user && articleData.isLikedByCurrentUser !== undefined) {
      setIsLiked(articleData.isLikedByCurrentUser);
    } else {
      setIsLiked(false);
    }
  }, [user, articleData.isLikedByCurrentUser]);

  // Fetch navigation articles when article data changes
  useEffect(() => {
    let isActive = true; // Prevent race conditions
    let timeoutId = null; // Safety timeout
    
    const loadNavigationArticles = async () => {
      if (articleData && articleData.category) {
        console.log('Loading navigation articles for:', articleData.id || articleData._id, 'category:', articleData.category);
        setNavigationLoading(true);
        
        // Small delay to allow category pages to populate cache first
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Safety timeout to prevent stuck loading state
        timeoutId = setTimeout(() => {
          if (isActive) {
            console.warn('Navigation loading timeout reached, clearing loading state');
            setNavigationArticles({ previousArticle: null, nextArticle: null });
            setNavigationLoading(false);
          }
        }, 5000); // Reduced to 5 seconds since we have better error handling now
        
        try {
          const navArticles = await getNavigationArticles(articleData);
          
          // Clear timeout since we got a response
          if (timeoutId) {
            clearTimeout(timeoutId);
            timeoutId = null;
          }
          
          // Only update state if this effect is still active
          if (isActive) {
            console.log('Setting navigation articles:', {
              hasPrevious: !!navArticles.previousArticle,
              hasNext: !!navArticles.nextArticle,
              previousTitle: navArticles.previousArticle?.translations?.en?.title?.substring(0, 30),
              nextTitle: navArticles.nextArticle?.translations?.en?.title?.substring(0, 30)
            });
            setNavigationArticles(navArticles);
            setNavigationLoading(false);
          }
        } catch (error) {
          console.error('Error loading navigation articles:', error);
          
          // Clear timeout
          if (timeoutId) {
            clearTimeout(timeoutId);
            timeoutId = null;
          }
          
          if (isActive) {
            // Even on error, clear loading state
            setNavigationArticles({ previousArticle: null, nextArticle: null });
            setNavigationLoading(false);
          }
        }
      } else {
        console.log('No article data or category, clearing navigation loading');
        setNavigationLoading(false);
      }
    };

    loadNavigationArticles();
    
    // Cleanup function to prevent race conditions
    return () => {
      isActive = false;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [articleData.id, articleData._id, articleData.category]);
  
  if (!articleData) {
    return <div>{t('No articles available')}</div>;
  }

  // Handle article like toggle
  const handleLikeToggle = async () => {
    if (!isAuthenticated) {
      // Could show auth modal or redirect to sign in
      return;
    }

    setLikeLoading(true);
    try {
      // Use _id for database articles or id for static articles
      const articleId = articleData._id || articleData.id;
      console.log('Attempting to like article:', { articleId, articleData });
      if (!articleId) {
        console.error('No article ID found:', articleData);
        return;
      }
      const response = await articleApi.toggleLike(articleId);
      
      // Update the like state based on backend response
      setIsLiked(response.data.isLiked);
      
      // Update article data with new like count and like status
      setArticleData(prev => ({
        ...prev,
        likes: {
          ...prev.likes,
          count: response.data.likes
        },
        isLikedByCurrentUser: response.data.isLiked
      }));
    } catch (error) {
      console.error('Failed to toggle like:', error);
    } finally {
      setLikeLoading(false);
    }
  };

  // Handle article unpublish (admin only)
  const handleUnpublish = async () => {
    if (!isAdmin) {
      return;
    }

    const confirmed = window.confirm(t('Are you sure you want to unpublish this article? It will be moved to drafts and no longer visible to users.'));
    if (!confirmed) {
      return;
    }

    try {
      const articleId = articleData._id || articleData.id;
      if (!articleId) {
        console.error('No article ID found:', articleData);
        return;
      }
      
      await articleApi.unpublish(articleId);
      
      // Show success message and redirect to home or admin dashboard
      alert(t('Article has been unpublished successfully'));
      navigate('/');
    } catch (error) {
      console.error('Failed to unpublish article:', error);
      alert(t('Failed to unpublish article. Please try again.'));
    }
  };

  // Get current language content or fallback to English
  const getCurrentLanguageContent = () => {
    const currentLang = i18n.language;
    if (article.translations && article.translations[currentLang]) {
      return article.translations[currentLang];
    }
    // Fallback to English if current language not available
    return article.translations?.en || {
      title: 'Untitled',
      content: [],
      excerpt: ''
    };
  };

  const localizedContent = getCurrentLanguageContent();

  // Get author name based on language
  const getAuthorName = () => {
    if (i18n.language === 'ar') {
      return 'صدقي بن حوالة';
    }
    return article.author?.name || article.author || 'Anonymous';
  };

  // Get category name based on language
  const getCategoryName = () => {
    const categoryTranslations = {
      'etoile-du-sahel': {
        en: 'Etoile Du Sahel',
        fr: 'Étoile Du Sahel', 
        ar: 'النجم الساحلي'
      },
      'the-beautiful-game': {
        en: 'The Beautiful Game',
        fr: 'Le Beau Jeu',
        ar: 'اللعبة الجميلة'
      },
      'all-sports-hub': {
        en: 'All Sports Hub',
        fr: 'Centre Omnisports',
        ar: 'مركز جميع الرياضات'
      }
    };

    return categoryTranslations[article.category]?.[i18n.language] || article.category;
  };

  // Check if current language is RTL
  const isRTL = i18n.language === 'ar';

  // Theme colors based on article category
  const themeColors = {
    'etoile-du-sahel': {
      light: 'bg-red-100 text-red-900 dark:bg-red-900 dark:text-red-100',
      border: 'border-red-900 dark:border-red-600',
      hover: 'hover:text-red-900 dark:hover:text-red-400',
      icon: 'text-red-500'
    },
    'the-beautiful-game': {
      light: 'bg-green-100 text-green-900 dark:bg-green-900 dark:text-green-100',
      border: 'border-green-900 dark:border-green-600',
      hover: 'hover:text-green-900 dark:hover:text-green-400',
      icon: 'text-green-500'
    },
    'all-sports-hub': {
      light: 'bg-purple-100 text-purple-900 dark:bg-purple-900 dark:text-purple-100',
      border: 'border-purple-900 dark:border-purple-600',
      hover: 'hover:text-purple-900 dark:hover:text-purple-400',
      icon: 'text-purple-500'
    },
    'archive': {
      light: 'bg-yellow-100 text-yellow-900 dark:bg-yellow-900 dark:text-yellow-100',
      border: 'border-yellow-900 dark:border-yellow-600',
      hover: 'hover:text-yellow-900 dark:hover:text-yellow-400',
      icon: 'text-yellow-500'
    },
    default: {
      light: 'bg-gray-100 text-gray-900 dark:bg-gray-900 dark:text-gray-100',
      border: 'border-gray-900 dark:border-gray-600',
      hover: 'hover:text-gray-900 dark:hover:text-gray-400',
      icon: 'text-gray-500'
    }
  };

  const theme = themeColors[article.category] || themeColors.default;

  // Function to render content blocks
  const renderContentBlock = (block, index) => {
    const { type, content, metadata = {} } = block;
    
    // Apply styling from metadata
    const blockStyle = {
      marginTop: metadata.style?.margins?.top ? `${metadata.style.margins.top}px` : undefined,
      marginBottom: metadata.style?.margins?.bottom ? `${metadata.style.margins.bottom}px` : undefined,
      color: metadata.style?.textColor || undefined,
      backgroundColor: metadata.style?.backgroundColor || undefined,
    };

    // Apply text alignment
    const alignmentClass = metadata.alignment ? `text-${metadata.alignment}` : '';

    switch (type) {
      case 'paragraph':
        return (
          <div 
            key={index} 
            className={`mb-6 text-gray-700 dark:text-gray-300 leading-relaxed ${alignmentClass}`}
            style={blockStyle}
            dangerouslySetInnerHTML={{ __html: content }}
          />
        );

      case 'heading':
        const HeadingTag = `h${metadata.level || 2}`;
        const headingSize = {
          2: 'text-3xl',
          3: 'text-2xl', 
          4: 'text-xl'
        }[metadata.level || 2];
        
        return (
          <HeadingTag 
            key={index}
            className={`${headingSize} font-serif font-bold text-gray-900 dark:text-white mt-10 mb-6 ${alignmentClass}`}
            style={blockStyle}
            dangerouslySetInnerHTML={{ __html: content }}
          />
        );

      case 'quote':
        // Get border color based on theme
        const borderColor = theme.border.includes('red') ? '#dc2626' :
                           theme.border.includes('green') ? '#16a34a' :
                           theme.border.includes('purple') ? '#9333ea' :
                           theme.border.includes('yellow') ? '#ca8a04' : '#374151';
        
        const quoteStyle = {
          ...blockStyle,
          [isRTL ? 'borderRight' : 'borderLeft']: `4px solid ${borderColor}`
        };
        
        return (
          <blockquote 
            key={index}
            className={`relative ${isRTL ? 'pr-8 pl-4' : 'pl-8 pr-4'} py-4 my-8 italic text-lg ${alignmentClass}`}
            style={quoteStyle}
          >
            <div 
              className="text-gray-800 dark:text-gray-200"
              dangerouslySetInnerHTML={{ __html: content }}
            />
            {metadata.source && (
              <footer className="text-gray-600 dark:text-gray-400 mt-3 not-italic text-base">
                — {metadata.source}
              </footer>
            )}
          </blockquote>
        );

      case 'image':
        const imageUrl = metadata.images?.[0]?.url || content;
        const backendUrl = process.env.REACT_APP_API_URL?.replace('/api', '') || 'http://localhost:5000';
        
        // Construct full image URL if it's a relative path
        const fullImageUrl = imageUrl?.startsWith('http') ? imageUrl : 
                            imageUrl?.startsWith('/') ? `${backendUrl}${imageUrl}` : imageUrl;
        
        return (
          <figure key={index} className={`my-8 ${alignmentClass}`} style={blockStyle}>
            <img
              src={fullImageUrl}
              alt={metadata.caption || metadata.images?.[0]?.caption || ''}
              className="w-full max-w-4xl mx-auto rounded-lg shadow-lg"
              style={{
                objectFit: 'cover',
                height: metadata.images?.[0]?.size === 'small' ? '300px' : 
                       metadata.images?.[0]?.size === 'large' ? '600px' : 'auto'
              }}

            />
            {(metadata.caption || metadata.images?.[0]?.caption) && (
              <figcaption className="mt-3 text-center text-sm text-gray-600 dark:text-gray-400">
                {metadata.caption || metadata.images[0].caption}
              </figcaption>
            )}
          </figure>
        );

      case 'image-group':
        const images = metadata.images || [];
        if (images.length === 0) return null;
        
        const backendUrlGroup = process.env.REACT_APP_API_URL?.replace('/api', '') || 'http://localhost:5000';

        return (
          <figure key={index} className={`my-8 ${alignmentClass}`} style={blockStyle}>
            <div className={`grid gap-4 ${
              images.length === 1 ? 'grid-cols-1' :
              images.length === 2 ? 'grid-cols-1 md:grid-cols-2' :
              'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'
            }`}>
              {images.map((image, imgIndex) => {
                const fullImageUrl = image.url?.startsWith('http') ? image.url : 
                                   image.url?.startsWith('/') ? `${backendUrlGroup}${image.url}` : image.url;
                
                return (
                  <div key={imgIndex} className="space-y-2">
                    <img
                      src={fullImageUrl}
                      alt={image.caption || `Image ${imgIndex + 1}`}
                      className="w-full object-cover rounded-lg shadow-lg"
                      style={{
                        height: image.size === 'small' ? '200px' :
                               image.size === 'large' ? '400px' : '300px'
                      }}

                    />
                    {image.caption && (
                      <p className="text-sm text-gray-600 dark:text-gray-400 text-center">
                        {image.caption}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </figure>
        );

      case 'list':
        const ListTag = metadata.listType === 'numbered' ? 'ol' : 'ul';
        const listClass = metadata.listType === 'numbered' ? 'list-decimal' : 'list-disc';
        
        return (
          <ListTag 
            key={index}
            className={`${listClass} ${isRTL ? 'mr' : 'ml'}-6 my-6 space-y-2 text-gray-700 dark:text-gray-300 ${alignmentClass}`}
            style={blockStyle}
          >
            <div dangerouslySetInnerHTML={{ __html: content }} />
          </ListTag>
        );

      // Backward compatibility for old content structure
      case 'subheading':
        return (
          <h2 
            key={index}
            className={`text-2xl font-serif font-bold text-gray-900 dark:text-white mt-8 mb-4 ${alignmentClass}`}
            style={blockStyle}
          >
            {content}
          </h2>
        );

      default:
        // Fallback: treat unknown types as paragraphs with HTML content
        return (
          <div 
            key={index}
            className={`mb-6 text-gray-700 dark:text-gray-300 leading-relaxed ${alignmentClass}`}
            style={blockStyle}
            dangerouslySetInnerHTML={{ __html: content }}
          />
        );
    }
  };

  return (
    <>
      <article className={`max-w-4xl mx-auto bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden border-l-4`} 
        style={{borderLeftColor: theme.icon.includes('red') ? '#ef4444' : theme.icon.includes('green') ? '#22c55e' : theme.icon.includes('purple') ? '#a855f7' : theme.icon.includes('yellow') ? '#eab308' : '#6b7280'}} 
        dir={isRTL ? 'rtl' : 'ltr'}>
      {/* Hero Image */}
      <div className="relative h-[40vh] md:h-[50vh]">
        <img
          src={article.image}
          alt={localizedContent.title}
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
      </div>

      {/* Article Content */}
      <div className={`px-6 py-8 md:px-10 ${isRTL ? 'text-right' : 'text-left'}`}>
        {/* Article Header */}
        <div className={`mb-8 pb-6 border-b ${theme.border} border-opacity-20`}>
          <div className={`flex items-center gap-4 mb-4 ${isRTL ? 'flex-row-reverse justify-end' : 'justify-start'}`}>
            <span className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium ${theme.light}`}>
              <FiBookmark className={theme.icon} />
              {getCategoryName()}
            </span>
            <span className="text-gray-600 dark:text-gray-400">
              {article.date}
            </span>
          </div>
          <h1 className="text-3xl md:text-4xl lg:text-5xl font-serif font-bold text-gray-900 dark:text-white mb-4">
            {localizedContent.title}
          </h1>
          <div className={`flex items-center gap-4 ${isRTL ? 'flex-row-reverse justify-end' : 'justify-start'}`} style={{ direction: isRTL ? 'rtl' : 'ltr' }}>
            <img
              src={article.authorImage || 'https://via.placeholder.com/40'}
              alt={getAuthorName()}
              className={`w-12 h-12 rounded-full object-cover object-center border-2 ${theme.border}`}
            />
            <span className="font-medium text-gray-900 dark:text-white">
              {getAuthorName()}
            </span>
          </div>
        </div>

        {/* Article Body */}
        <div className={`prose prose-lg dark:prose-invert max-w-none ${isRTL ? 'prose-rtl' : ''}`}>
          {localizedContent.content && localizedContent.content.length > 0 ? (
            localizedContent.content.map((block, index) => (
              <div key={index}>
                {renderContentBlock(block, index)}
                {/* Themed separator after every 3rd content block */}
                {(index + 1) % 3 === 0 && index < localizedContent.content.length - 1 && (
                  <div className={`my-8 border-t ${theme.border} opacity-30`}></div>
                )}
              </div>
            ))
          ) : (
            <div className="text-gray-600 dark:text-gray-400 italic">
              {t('No content available for this article.')}
            </div>
          )}
        </div>

        {/* Article Footer with Tags */}
        <div className={`mt-12 pt-6 border-t ${theme.border} opacity-30`}>
          {article.tags && article.tags.length > 0 && (
            <div className="mb-6">
              <h3 className={`text-lg font-semibold mb-4 ${theme.icon} ${isRTL ? 'text-right' : 'text-left'}`}>
                {t('Tags')}
              </h3>
              <div className={`flex flex-wrap gap-2 ${isRTL ? 'flex-row-reverse' : ''}`}>
                {article.tags.map((tag, index) => (
                  <span
                    key={index}
                    className={`px-3 py-1 rounded-full text-sm font-medium transition-colors cursor-pointer ${theme.light} ${theme.hover}`}
                  >
                    #{tag}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

      </div>

    </article>

    {/* Article Navigation */}
    <ArticleNavigation 
      currentArticle={articleData}
      previousArticle={navigationArticles.previousArticle}
      nextArticle={navigationArticles.nextArticle}
      loading={navigationLoading}
    />

    {/* Admin Buttons */}
    {isAdmin && (
      <>
        {/* Admin Edit Button */}
        <div className={`fixed bottom-8 z-50 ${isRTL ? 'left-28' : 'right-28'}`}>
          <button
            onClick={() => navigate(`/admin/edit-article/${articleData._id || articleData.id}`)}
            className="group relative w-16 h-16 bg-white dark:bg-gray-800 rounded-full shadow-2xl transition-all duration-300 transform hover:scale-110 hover:shadow-xl"
            style={{
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)'
            }}
          >
            <div 
              className="absolute inset-0 rounded-full opacity-0 group-hover:opacity-20 transition-opacity duration-300"
              style={{
                background: `linear-gradient(45deg, ${theme.icon.includes('red') ? '#dc2626, #ef4444' : theme.icon.includes('green') ? '#16a34a, #22c55e' : theme.icon.includes('purple') ? '#9333ea, #a855f7' : theme.icon.includes('yellow') ? '#ca8a04, #eab308' : '#374151, #6b7280'})`
              }}
            ></div>
            
            <FiEdit3 className={`w-8 h-8 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 transition-all duration-300 text-gray-600 dark:text-gray-300 group-hover:${theme.icon}`} />
            
            {/* Tooltip */}
            <div 
              className={`absolute bottom-full ${isRTL ? 'left-0' : 'right-0'} mb-2 px-3 py-1 text-white text-sm rounded-lg opacity-0 group-hover:opacity-100 transition-all duration-200 whitespace-nowrap`}
              style={{
                backgroundColor: theme.icon.includes('red') ? '#dc2626' : theme.icon.includes('green') ? '#16a34a' : theme.icon.includes('purple') ? '#9333ea' : theme.icon.includes('yellow') ? '#ca8a04' : '#374151'
              }}
            >
              {t('Edit Article')}
              <div 
                className={`absolute top-full ${isRTL ? 'left-4' : 'right-4'} w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent`}
                style={{
                  borderTopColor: theme.icon.includes('red') ? '#dc2626' : theme.icon.includes('green') ? '#16a34a' : theme.icon.includes('purple') ? '#9333ea' : theme.icon.includes('yellow') ? '#ca8a04' : '#374151'
                }}
              ></div>
            </div>
          </button>
        </div>

        {/* Admin Delete Button */}
        <div className={`fixed bottom-8 z-50 ${isRTL ? 'left-48' : 'right-48'}`}>
          <button
            onClick={handleUnpublish}
            className="group relative w-16 h-16 bg-white dark:bg-gray-800 rounded-full shadow-2xl transition-all duration-300 transform hover:scale-110 hover:shadow-xl"
            style={{
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)'
            }}
          >
            <div 
              className="absolute inset-0 rounded-full opacity-0 group-hover:opacity-20 transition-opacity duration-300"
              style={{
                background: `linear-gradient(45deg, ${theme.icon.includes('red') ? '#dc2626, #ef4444' : theme.icon.includes('green') ? '#16a34a, #22c55e' : theme.icon.includes('purple') ? '#9333ea, #a855f7' : theme.icon.includes('yellow') ? '#ca8a04, #eab308' : '#374151, #6b7280'})`
              }}
            ></div>
            
            <FiTrash2 className={`w-8 h-8 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 transition-all duration-300 text-gray-600 dark:text-gray-300 group-hover:${theme.icon}`} />
            
            {/* Tooltip */}
            <div 
              className={`absolute bottom-full ${isRTL ? 'left-0' : 'right-0'} mb-2 px-3 py-1 text-white text-sm rounded-lg opacity-0 group-hover:opacity-100 transition-all duration-200 whitespace-nowrap`}
              style={{
                backgroundColor: theme.icon.includes('red') ? '#dc2626' : theme.icon.includes('green') ? '#16a34a' : theme.icon.includes('purple') ? '#9333ea' : theme.icon.includes('yellow') ? '#ca8a04' : '#374151'
              }}
            >
              {t('Unpublish Article')}
              <div 
                className={`absolute top-full ${isRTL ? 'left-4' : 'right-4'} w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent`}
                style={{
                  borderTopColor: theme.icon.includes('red') ? '#dc2626' : theme.icon.includes('green') ? '#16a34a' : theme.icon.includes('purple') ? '#9333ea' : theme.icon.includes('yellow') ? '#ca8a04' : '#374151'
                }}
              ></div>
            </div>
          </button>
        </div>
      </>
    )}

    {/* Floating Like Button */}
    <div className={`fixed bottom-8 z-50 ${isRTL ? 'left-8' : 'right-8'} group`}>
      <button
        onClick={handleLikeToggle}
        disabled={likeLoading}
        className={`relative w-16 h-16 bg-white dark:bg-gray-800 rounded-full shadow-2xl transition-all duration-300 transform hover:scale-110 hover:shadow-xl ${likeLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
        style={{
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)'
        }}
      >
        <div 
          className="absolute inset-0 rounded-full opacity-0 group-hover:opacity-20 transition-opacity duration-300"
          style={{
            background: `linear-gradient(45deg, ${theme.icon.includes('red') ? '#dc2626, #ef4444' : theme.icon.includes('green') ? '#16a34a, #22c55e' : theme.icon.includes('purple') ? '#9333ea, #a855f7' : theme.icon.includes('yellow') ? '#ca8a04, #eab308' : '#374151, #6b7280'})`
          }}
        ></div>
        
        <FiHeart 
          className={`w-8 h-8 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 transition-all duration-300 ${
            isLiked
              ? `${theme.icon} fill-current`
              : `text-gray-600 dark:text-gray-300 group-hover:${theme.icon}`
          }`}
        />
      </button>
      
      {/* Tooltip */}
      <div 
        className={`absolute bottom-full ${isRTL ? 'left-0' : 'right-0'} mb-2 px-3 py-1 text-white text-sm rounded-lg opacity-0 group-hover:opacity-100 transition-all duration-200 whitespace-nowrap pointer-events-none`}
        style={{
          backgroundColor: theme.icon.includes('red') ? '#dc2626' : theme.icon.includes('green') ? '#16a34a' : theme.icon.includes('purple') ? '#9333ea' : theme.icon.includes('yellow') ? '#ca8a04' : '#374151'
        }}
      >
        {isLiked ? t('Unlike') : t('Like this article')}
        <div 
          className={`absolute top-full ${isRTL ? 'left-4' : 'right-4'} w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent`}
          style={{
            borderTopColor: theme.icon.includes('red') ? '#dc2626' : theme.icon.includes('green') ? '#16a34a' : theme.icon.includes('purple') ? '#9333ea' : theme.icon.includes('yellow') ? '#ca8a04' : '#374151'
          }}
        ></div>
      </div>
    </div>

    {/* Comments Section */}
    <CommentsSection 
      articleId={articleData._id || articleData.id} 
      category={articleData.category}
      theme={theme}
      onCommentCountChange={(newCount) => {
        // Update local article state
        setArticleData(prev => ({
          ...prev,
          comments: newCount
        }));
        
        // Update global article cache to notify all category pages
        const articleId = articleData._id || articleData.id;
        if (articleId) {
          updateArticleCommentCount(articleId, newCount);
        }
      }}
    />

    {/* Newsletter Section */}
    <div className="mt-12">
      <Newsletter variant={articleData.category} />
    </div>
  </>
  );
}

export default Article; 