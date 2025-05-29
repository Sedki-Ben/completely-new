import article1Image from '../assets/images/articles/champions-league.jpg';
import article2Image from '../assets/images/articles/world-cup.jpg';
import article3Image from '../assets/images/articles/premier-league.png';

export const articles = [
    {
      id: 1,
    translations: {
      en: {
      title: "Tactical Analysis: Champions League Final",
        excerpt: "Breaking down the key moments that decided Europe's biggest game",
        content: [
          {
            type: "paragraph",
            content: "The Champions League final between Manchester City and Inter Milan showcased a fascinating tactical battle. Pep Guardiola's side demonstrated their positional play mastery, while Simone Inzaghi's Inter proved resilient and tactically disciplined."
          },
          {
            type: "subheading",
            content: "City's Build-up Pattern"
          },
          {
            type: "paragraph",
            content: "Manchester City's build-up play was characterized by their typical 3-2 structure, with John Stones often stepping into midfield. This created numerical superiority in the middle third and allowed them to bypass Inter's first line of pressure."
          },
          {
            type: "quote",
            content: "In football, the hardest thing is not to play well but to know what to do at each moment.",
            author: "Johan Cruyff"
          }
        ]
      },
      fr: {
        title: "Analyse Tactique : Finale de la Ligue des Champions",
        excerpt: "Décryptage des moments clés qui ont décidé du plus grand match d'Europe",
        content: [
          {
            type: "paragraph",
            content: "La finale de l'UEFA Champions League entre Manchester City et l'Inter Milan a été un chef-d'œuvre d'adaptation tactique et de planification stratégique. Dans cette analyse, nous allons décortiquer les moments clés et les décisions tactiques qui ont finalement décidé du sort du match le plus prestigieux du football européen."
          },
          {
            type: "subheading",
            content: "Configuration Tactique d'Avant-Match"
          },
          {
            type: "paragraph",
            content: "Pep Guardiola a opté pour sa formation 4-3-3 préférée, mais avec une subtile variation. Le poste de faux neuf a été utilisé pour créer la confusion dans les lignes défensives de l'Inter, tandis que les arrières latéraux avaient la liberté de se replier en position de milieu de terrain en phase de possession."
          },
          {
            type: "quote",
            content: "La beauté du football réside dans sa complexité tactique - c'est comme échecs, mais avec 22 pièces en mouvement simultanément.",
            author: "Pep Guardiola"
          },
          {
            type: "paragraph",
            content: "L'Inter Milan, sous la direction de Simone Inzaghi, s'est installé en leur traditionnel 3-5-2 formation, avec les ailiers-défenseurs jouant un rôle crucial dans les transitions tant défensives que offensives. Ce système a été conçu pour annihiler les menaces latérales de Manchester City tout en maintenant la capacité de contrer efficacement les contre-attaques."
          }
        ]
      },
      ar: {
        title: "تحليل تكتيكي: نهائي دوري أبطال أوروبا",
        excerpt: "تحليل اللحظات المفصلية التي حسمت أكبر مباراة في أوروبا",
        content: [
          {
            type: "paragraph",
            content: "أظهر نهائي دوري أبطال أوروبا بين مانشستر سيتي وإنتر ميلان معركة تكتيكية مثيرة. أظهر فريق بيب غوارديولا براعته في اللعب الموقعي، بينما أثبت إنتر بقيادة سيموني إنزاجي صلابته وانضباطه التكتيكي."
          },
          {
            type: "subheading",
            content: "نمط بناء الهجمات لدى السيتي"
          },
          {
            type: "paragraph",
            content: "تميز أسلوب بناء الهجمات لدى مانشستر سيتي بتشكيلهم المعتاد 3-2، مع تقدم جون ستونز غالباً إلى خط الوسط. هذا خلق تفوقاً عددياً في الثلث الأوسط وسمح لهم بتجاوز خط الضغط الأول لإنتر."
          },
          {
            type: "quote",
            content: "في كرة القدم، الشيء الأصعب ليس أن تلعب بشكل جيد ولكن أن تعرف ماذا تفعل في كل لحظة.",
            author: "يوهان كرويف"
          }
        ]
      }
    },
      author: "John Doe",
    authorImage: "https://via.placeholder.com/40",
    date: "January 15, 2024",
    image: article1Image,
    category: "analysis",
    likes: 245,
    comments: 58
  },
  {
    id: 2,
    translations: {
      en: {
        title: "World Cup 2026: Early Predictions",
        excerpt: "Which teams are shaping up to be contenders in North America",
        content: [
          {
            type: "paragraph",
            content: "As we look ahead to the 2026 FIFA World Cup, the first to be hosted across three nations - the United States, Mexico, and Canada - several teams are already emerging as strong contenders. This expanded tournament format will see 48 teams compete for football's ultimate prize."
          },
          {
            type: "subheading",
            content: "The Favorites"
          },
          {
            type: "paragraph",
            content: "Brazil, with their emerging young talents and traditional flair, remains a top contender. The European powerhouses - France, Germany, and Spain - are rebuilding with exciting young prospects. Argentina, the current champions, will be looking to defend their title with a mix of experience and youth."
          },
          {
            type: "quote",
            content: "The 48-team format will give more nations a chance to dream, but it will also make the journey to the final even more challenging.",
            author: "Carlos Alberto Parreira"
          }
        ]
      }
    },
    author: "Jane Smith",
    authorImage: "https://via.placeholder.com/40",
    date: "January 10, 2024",
    image: article2Image,
    category: "stories",
    likes: 189,
    comments: 42
  },
  {
    id: 3,
    translations: {
      en: {
        title: "Premier League Title Race Heats Up",
        excerpt: "Why this might be the closest title race in recent memory",
        content: [
          {
            type: "paragraph",
            content: "This season's Premier League title race is shaping up to be one of the most competitive in recent history. With multiple teams still in contention at the halfway point, we're analyzing the factors that could determine the eventual champion."
          },
          {
            type: "subheading",
            content: "The Contenders"
          },
          {
            type: "paragraph",
            content: "Arsenal has shown remarkable consistency, while Manchester City's experience in title races cannot be underestimated. Liverpool's resurgence and Aston Villa's surprise challenge have added extra spice to what is becoming an enthralling battle for supremacy."
          },
          {
            type: "quote",
            content: "The Premier League is like a marathon where you also have to sprint every 100 meters.",
            author: "Jurgen Klopp"
          }
        ]
      }
    },
    author: "Mike Johnson",
    date: "January 5, 2024",
    authorImage: "https://via.placeholder.com/40",
    image: article3Image,
    category: "analysis",
    excerpt: "Why this might be the closest title race in recent memory",
    likes: 312,
    comments: 76
  },
  {
    id: 4,
    translations: {
      en: {
        title: "The Evolution of Total Football",
        excerpt: "How Dutch innovation changed football forever",
        content: [
          {
            type: "paragraph",
            content: "Total Football, pioneered by the Dutch in the 1970s, revolutionized how we think about tactical systems and player roles. This comprehensive analysis explores its evolution and modern interpretations."
          }
        ]
      }
    },
    author: "David Wilson",
    authorImage: "https://via.placeholder.com/40",
    date: "December 15, 2023",
    image: article1Image,
    category: "notable",
    likes: 423,
    comments: 89
  },
  {
    id: 6,
    translations: {
      en: {
        title: "The Rise of African Football: A New Era Begins",
        excerpt: "How African nations are reshaping the global football landscape",
        content: [
          {
            type: "paragraph",
            content: "African football is experiencing an unprecedented renaissance, with more players than ever making their mark in Europe's top leagues. This transformation is not just about individual talent, but represents a broader shift in how African nations approach football development."
          },
          {
            type: "subheading",
            content: "Youth Development Revolution"
          },
          {
            type: "paragraph",
            content: "Countries like Senegal and Morocco have invested heavily in youth academies and infrastructure, creating sustainable pathways for young talents. The results are already showing, with African teams making historic achievements in international competitions."
          },
          {
            type: "quote",
            content: "African football is no longer just about raw talent - it's about sophisticated development systems and professional structures.",
            author: "Sadio Mané"
          },
          {
            type: "paragraph",
            content: "With improved coaching education and modern facilities, the continent is witnessing a transformation that could reshape the global football hierarchy in the coming decades."
          }
        ]
      }
    },
    author: "Michael Obi",
    authorImage: "https://via.placeholder.com/40",
    date: "January 20, 2024",
    image: article2Image,
    category: "stories",
    likes: 178,
    comments: 45
  },
  {
    id: 7,
    translations: {
      en: {
        title: "Football's Digital Revolution: How Technology is Changing the Game",
        excerpt: "From VAR to AI: The technological transformation of football",
        content: [
          {
            type: "paragraph",
            content: "The beautiful game is undergoing a digital transformation that extends far beyond VAR. From AI-powered performance analysis to blockchain-based fan engagement, technology is reshaping how football is played, watched, and experienced."
          },
          {
            type: "subheading",
            content: "The Data Revolution"
          },
          {
            type: "paragraph",
            content: "Clubs are now employing data scientists and machine learning experts to gain competitive advantages. Every aspect of the game is being quantified and analyzed, from player movements to tactical patterns."
          },
          {
            type: "quote",
            content: "The future of football will be shaped by those who best harness the power of data and technology.",
            author: "Arsène Wenger"
          },
          {
            type: "paragraph",
            content: "This technological revolution is democratizing football analysis and creating new opportunities for clubs of all sizes to compete at the highest level."
          }
        ]
      }
    },
    author: "Sarah Chen",
    authorImage: "https://via.placeholder.com/40",
    date: "January 18, 2024",
    image: article1Image,
    category: "stories",
    likes: 203,
    comments: 67
  },
  {
    id: 8,
    translations: {
      en: {
        title: "The Modern Pressing Game: A Tactical Deep Dive",
        excerpt: "Analyzing how elite teams have perfected the art of pressing",
        content: [
          {
            type: "paragraph",
            content: "High pressing has become a defining characteristic of modern football, but its implementation varies significantly between teams. This analysis breaks down the different pressing styles employed by elite clubs and their effectiveness."
          },
          {
            type: "subheading",
            content: "Pressing Triggers"
          },
          {
            type: "paragraph",
            content: "Understanding when and how to press is crucial. Teams like Manchester City and Liverpool have developed sophisticated pressing triggers that allow them to win the ball in advantageous positions while maintaining defensive stability."
          },
          {
            type: "quote",
            content: "Pressing is not about running more, it's about running smarter.",
            author: "Jürgen Klopp"
          },
          {
            type: "paragraph",
            content: "The success of a pressing system depends on collective understanding and precise timing. We examine how top teams coordinate their press and adapt it to different opponents."
          }
        ]
      }
    },
    author: "Thomas Mueller",
    authorImage: "https://via.placeholder.com/40",
    date: "January 16, 2024",
    image: article3Image,
    category: "analysis",
    likes: 289,
    comments: 82
  },
  {
    id: 9,
    translations: {
      en: {
        title: "The False Nine Evolution: From Messi to Haaland",
        excerpt: "How the role of the striker has transformed in modern football",
        content: [
          {
            type: "paragraph",
            content: "The false nine position has evolved dramatically from Messi's interpretation to modern adaptations by players like Haaland. This analysis explores the tactical implications and future of this revolutionary role."
          }
        ]
      }
    },
    author: "Marco Rossi",
    authorImage: "https://via.placeholder.com/40",
    date: "January 22, 2024",
    image: article1Image,
    category: "analysis",
    likes: 156,
    comments: 42
  },
  {
    id: 10,
    translations: {
      en: {
        title: "Data Analytics in Football Scouting",
        excerpt: "How big data is revolutionizing talent identification",
        content: [
          {
            type: "paragraph",
            content: "Modern football scouting combines traditional methods with sophisticated data analytics. We explore how clubs are using AI and machine learning to discover the next generation of stars."
          }
        ]
      }
    },
    author: "Dr. James Wilson",
    authorImage: "https://via.placeholder.com/40",
    date: "January 21, 2024",
    image: article2Image,
    category: "analysis",
    likes: 189,
    comments: 53
  },
  {
    id: 11,
    translations: {
      en: {
        title: "The Art of Set-Piece Design",
        excerpt: "Breaking down the most innovative free-kick and corner routines",
        content: [
          {
            type: "paragraph",
            content: "Set-pieces can be the difference in tight matches. This analysis examines the most creative and effective routines from top teams across Europe."
          }
        ]
      }
    },
    author: "Carlos Mendoza",
    authorImage: "https://via.placeholder.com/40",
    date: "January 19, 2024",
    image: article3Image,
    category: "analysis",
    likes: 234,
    comments: 67
  },
  {
    id: 12,
    translations: {
      en: {
        title: "Defensive Transitions in Modern Football",
        excerpt: "How top teams organize their counter-press",
        content: [
          {
            type: "paragraph",
            content: "The moments immediately after losing possession are crucial in modern football. We analyze how elite teams structure their defensive transitions and counter-pressing mechanisms."
          }
        ]
      }
    },
    author: "Hans Schmidt",
    authorImage: "https://via.placeholder.com/40",
    date: "January 17, 2024",
    image: article1Image,
    category: "analysis",
    likes: 167,
    comments: 38
  },
  {
    id: 13,
    translations: {
      en: {
        title: "Goalkeeper Evolution: The Modern Sweeper-Keeper",
        excerpt: "Analyzing the expanded role of goalkeepers in build-up play",
        content: [
          {
            type: "paragraph",
            content: "Modern goalkeepers are expected to be proficient with their feet and participate in build-up play. We examine how this evolution has changed football tactics."
          }
        ]
      }
    },
    author: "Peter van der Berg",
    authorImage: "https://via.placeholder.com/40",
      date: "January 15, 2024",
    image: article2Image,
    category: "analysis",
    likes: 198,
    comments: 45
  },
  {
    id: 14,
    translations: {
      en: {
        title: "From Refugee to Champion: The Inspiring Journey of Mohamed Ali",
        excerpt: "A story of perseverance, hope, and football as a universal language",
        content: [
          {
            type: "paragraph",
            content: "Mohamed Ali's journey from a refugee camp to professional football is a testament to the power of dreams and determination."
          }
        ]
      }
    },
    author: "Sofia Martinez",
    authorImage: "https://via.placeholder.com/40",
    date: "January 21, 2024",
    image: article3Image,
    category: "stories",
    likes: 312,
    comments: 89
  },
  {
    id: 15,
    translations: {
      en: {
        title: "The Sunday League Revolution",
        excerpt: "How grassroots football is embracing technology and professionalism",
        content: [
          {
            type: "paragraph",
            content: "Amateur football is undergoing a transformation with apps, tracking technology, and professional coaching methods becoming increasingly accessible."
          }
        ]
      }
    },
    author: "Tom Richards",
    authorImage: "https://via.placeholder.com/40",
    date: "January 20, 2024",
    image: article1Image,
    category: "stories",
    likes: 145,
    comments: 34
  },
  {
    id: 16,
    translations: {
      en: {
        title: "Football's Mental Health Revolution",
        excerpt: "How clubs are prioritizing player wellbeing beyond the pitch",
        content: [
          {
            type: "paragraph",
            content: "Professional football is finally addressing mental health openly, with clubs implementing comprehensive support systems for their players."
          }
        ]
      }
    },
    author: "Dr. Emma Thompson",
    authorImage: "https://via.placeholder.com/40",
    date: "January 19, 2024",
    image: article2Image,
    category: "stories",
    likes: 267,
    comments: 73
  },
  {
    id: 17,
    translations: {
      en: {
        title: "The Last Amateur Club",
        excerpt: "How one historic club maintains its amateur status in the professional era",
        content: [
          {
            type: "paragraph",
            content: "In an age of commercialization, one historic club continues to uphold amateur values while competing against professional teams."
          }
        ]
      }
    },
    author: "George Hamilton",
    authorImage: "https://via.placeholder.com/40",
    date: "January 18, 2024",
    image: article3Image,
    category: "stories",
    likes: 189,
    comments: 45
  },
  {
    id: 18,
    translations: {
      en: {
        title: "Football's Climate Challenge",
        excerpt: "How clubs are adapting to environmental concerns",
        content: [
          {
            type: "paragraph",
            content: "From solar-powered stadiums to carbon-neutral travel, football clubs are taking innovative steps to address climate change."
          }
        ]
      }
    },
    author: "Lisa Green",
    authorImage: "https://via.placeholder.com/40",
    date: "January 17, 2024",
    image: article1Image,
    category: "stories",
    likes: 223,
    comments: 58
  }
];

export const getArticleById = (id) => {
  return articles.find(article => article.id === parseInt(id));
};

export const getArticlesByCategory = (category) => {
  return articles.filter(article => article.category === category);
};

export const getAllArticles = () => {
  return articles;
};

// Helper function to get article content in the current language
export const getLocalizedArticleContent = (article, language = 'en') => {
  if (!article?.translations?.[language]) {
    // Fallback to English if translation not available
    return article?.translations?.['en'] || null;
  }
  return article.translations[language];
};