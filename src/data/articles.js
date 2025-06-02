import article1Image from '../assets/images/articles/champions-league.jpg';
import article2Image from '../assets/images/articles/world-cup.jpg';
import article3Image from '../assets/images/articles/premier-league.png';
import sedkiImage from '../assets/images/bild3.jpg';

export const categoryTranslations = {
  'etoile-du-sahel': { 
    en: 'Etoile Du Sahel', 
    fr: 'Étoile du Sahel', 
    ar: 'النجم الساحلي' 
  },
  'the-beautiful-game': { 
    en: 'The Beautiful Game', 
    fr: 'Le Beau Jeu', 
    ar: 'اللعبة الجميلة' 
  },
  'all-sports-hub': { 
    en: 'All-Sports Hub', 
    fr: 'Hub Tous Sports', 
    ar: 'مركز كل الرياضات' 
  },
  archive: { 
    en: 'Archive', 
    fr: 'Archives', 
    ar: 'الأرشيف' 
  },
};

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
      author: "Sedki B.Haouala",
    authorImage: sedkiImage,
    date: "January 15, 2024",
    image: article1Image,
    category: "etoile-du-sahel",
    likes: 0,
    comments: 0
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
      },
      fr: {
        title: "Analyse Prédictive : Coupe du Monde 2026",
        excerpt: "Quels sont les équipes qui se détachent pour être des candidats en Amérique du Nord",
        content: [
          {
            type: "paragraph",
            content: "Avec le Coupe du Monde 2026, le premier à être organisé à travers trois pays - les États-Unis, le Mexique et le Canada - plusieurs équipes sont déjà en train de se détacher pour être des candidats pour le plus grand prix du football."
          },
          {
            type: "subheading",
            content: "Les Favoris"
          },
          {
            type: "paragraph",
            content: "Le Brésil, avec ses jeunes talents émergents et son flair traditionnel, reste un candidat à la première place. Les puissances européennes - France, Allemagne et Espagne - reconstruisent avec des jeunes promesses excitantes. L'Argentine, les champions actuels, cherchera à défendre leur titre avec une combinaison d'expérience et de jeunesse."
          },
          {
            type: "quote",
            content: "Le format de 48 équipes donnera plus de nations la chance de rêver, mais cela rendra également le voyage jusqu'à la finale encore plus difficile.",
            author: "Carlos Alberto Parreira"
          }
        ]
      },
      ar: {
        title: "توقعات مبكرة لكأس العالم 2026",
        excerpt: "أي منتخبات تبدو جاهزة لتكون متنافسين في أمريكا الشمالية",
        content: [
          {
            type: "paragraph",
            content: "بينما ننظر إلى كأس العالم 2026، الأولى لتنظيمها عبر ثلاثة بلدان - الولايات المتحدة، المكسيك، وكندا - تبدو عدة أنديات من الممكن أن تكون متنافسين للميزانية الكبرى لكرة القدم."
          },
          {
            type: "subheading",
            content: "المفضلون"
          },
          {
            type: "paragraph",
            content: "البرازيل، مع خريجيه الشباب الذين يظهرون أثرهم التقليدي، يظل متنافسًا رئيسيًا. القوى الأوروبية - فرنسا وألمانيا وإسبانيا - تعيد بناء مع الفرص الشباب المثيرة للإثارة. آرجنتين، البطلون الحاليون، سيحاولون حماية عنوانهم مع خلط من الخبرة والشباب."
          },
          {
            type: "quote",
            content: "تنسيق الميزانية 48 فريقًا سيمنح المزيد من الدول فرصة للإحلال، ولكنه سيجعل الرحلة للنهائية أكثر صعوبة.",
            author: "كارلوس ألبرتو باريرا"
          }
        ]
      }
    },
    author: "Sedki B.Haouala",
    authorImage: sedkiImage,
    date: "January 10, 2024",
    image: article2Image,
    category: "the-beautiful-game",
    likes: 0,
    comments: 0
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
      },
      fr: {
        title: "Course à la Premier League",
        excerpt: "Pourquoi cette course pourrait être la plus compétitive de la mémoire récente",
        content: [
          {
            type: "paragraph",
            content: "La course à la Premier League est en train de prendre de l'ampleur pour être l'une des plus compétitive de l'histoire récente. Avec plusieurs équipes encore en lice à mi-parcours, nous analysons les facteurs qui pourraient déterminer le champion final."
          },
          {
            type: "subheading",
            content: "Les Contendeurs"
          },
          {
            type: "paragraph",
            content: "Arsenal a montré une consistance remarquable, tandis que l'expérience de Manchester City dans les courses pour le titre ne peut être sous-estimée. La résurrection de Liverpool et la surprise défi d'Aston Villa ont ajouté un arôme supplémentaire à ce qui devient une bataille acharnée pour la suprématie."
          },
          {
            type: "quote",
            content: "La Premier League est comme une course à pieds où vous devez également accélérer tous les 100 mètres.",
            author: "Jurgen Klopp"
          }
        ]
      },
      ar: {
        title: "منافسة الدوري الإنجليزي للممتلكات",
        excerpt: "لماذا قد تكون هذه المنافسة الأقرب في الذاكرة المؤخرة",
        content: [
          {
            type: "paragraph",
            content: "هذه المنافسة لدوري أبطال إنجلترا الموسم الحالي هي تبديد إلى أن تكون أكثر تنافسية في الذاكرة المؤخرة. مع العديد من الأندية الموجودة في المنافسة في منتصف المسار، نحن نحلل العوامل التي قد تحدد الفائز النهائي."
          },
          {
            type: "subheading",
            content: "المتنافسون"
          },
          {
            type: "paragraph",
            content: "أظهر آرسنال توافقًا مذهلًا بينما لم تكتفي خبرة مانشستر سيتي بالمنافسة للميزانية لأنها غير قابلة للتقدير. عادة مانشستر سيتي وتحدي الإستونفيلا الغامض قد أضاف حلوى إضافية لما يصبح معركة حارقة للسيطرة."
          },
          {
            type: "quote",
            content: "دوري إنجلترا هو مثل ماراثون حيث تحتاج أيضًا إلى السريع كل 100 مترًا.",
            author: "جورجن كلوب"
          }
        ]
      }
    },
    author: "Sedki B.Haouala",
    date: "January 5, 2024",
    authorImage: sedkiImage,
    image: article3Image,
    category: "etoile-du-sahel",
    excerpt: "Why this might be the closest title race in recent memory",
    likes: 0,
    comments: 0
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
      },
      fr: {
        title: "L'évolution du Football Total",
        excerpt: "Comment l'innovation néerlandaise a changé le football à jamais",
        content: [
          {
            type: "paragraph",
            content: "Le Football Total, initié par les Néerlandais dans les années 1970, a révolutionné la conception des systèmes tactiques et des rôles des joueurs. Cette analyse explore son évolution et ses interprétations modernes."
          }
        ]
      },
      ar: {
        title: "تطور الكرة الشاملة",
        excerpt: "كيف غيّر الابتكار الهولندي كرة القدم إلى الأبد",
        content: [
          {
            type: "paragraph",
            content: "كرة القدم الشاملة، التي ابتكرها الهولنديون في السبعينيات، غيرت طريقة التفكير في الأنظمة التكتيكية وأدوار اللاعبين. تستكشف هذه التحليل تطورها وتفسيراتها الحديثة."
          }
        ]
      }
    },
    author: "Sedki B.Haouala",
    authorImage: sedkiImage,
    date: "December 15, 2023",
    image: article1Image,
    category: "the-beautiful-game",
    likes: 0,
    comments: 0
  },
  {
    id: 5,
    translations: {
      en: {
        title: "The Renaissance of Italian Defending",
        excerpt: "How Serie A clubs are redefining defensive tactics in modern football",
        content: [
          {
            type: "paragraph",
            content: "Italian football has long been synonymous with defensive mastery. In recent years, Serie A clubs have blended traditional catenaccio with modern pressing and ball-playing defenders, creating a new defensive renaissance."
          },
          {
            type: "subheading",
            content: "Blending Old and New"
          },
          {
            type: "paragraph",
            content: "Teams like Inter and Juventus are combining disciplined back lines with proactive pressing, making Italian defenses some of the most difficult to break down in Europe."
          },
          {
            type: "quote",
            content: "Defending is an art, and Italy has always been its greatest gallery.",
            author: "Paolo Maldini"
          }
        ]
      },
      fr: {
        title: "La Renaissance de la Défense Italienne",
        excerpt: "Comment les clubs de Serie A redéfinissent la tactique défensive dans le football moderne",
        content: [
          {
            type: "paragraph",
            content: "Le football italien est depuis longtemps synonyme de maîtrise défensive. Ces dernières années, les clubs de Serie A ont fusionné le catenaccio traditionnel avec le pressing moderne et des défenseurs habiles avec le ballon, créant une nouvelle renaissance défensive."
          },
          {
            type: "subheading",
            content: "Mélange d'ancien et de nouveau"
          },
          {
            type: "paragraph",
            content: "Des équipes comme l'Inter et la Juventus combinent des lignes arrières disciplinées avec un pressing proactif, rendant les défenses italiennes parmi les plus difficiles à percer en Europe."
          },
          {
            type: "quote",
            content: "Défendre est un art, et l'Italie a toujours été sa plus grande galerie.",
            author: "Paolo Maldini"
          }
        ]
      },
      ar: {
        title: "نهضة الدفاع الإيطالي",
        excerpt: "كيف تعيد أندية الدوري الإيطالي تعريف التكتيك الدفاعي في كرة القدم الحديثة",
        content: [
          {
            type: "paragraph",
            content: "لطالما ارتبطت كرة القدم الإيطالية بالإتقان الدفاعي. في السنوات الأخيرة، مزجت أندية الدوري الإيطالي بين الكاتيناتشيو التقليدي والضغط الحديث والمدافعين المميزين بالكرة، مما خلق نهضة دفاعية جديدة."
          },
          {
            type: "subheading",
            content: "مزج القديم بالجديد"
          },
          {
            type: "paragraph",
            content: "تجمع فرق مثل إنتر ويوفنتوس بين خطوط دفاع منضبطة وضغط استباقي، مما يجعل الدفاعات الإيطالية من الأصعب اختراقها في أوروبا."
          },
          {
            type: "quote",
            content: "الدفاع فن، وإيطاليا كانت دائماً أعظم معارضه.",
            author: "باولو مالديني"
          }
        ]
      }
    },
    author: "Sedki B.Haouala",
    authorImage: sedkiImage,
    date: "December 20, 2023",
    image: article2Image,
    category: "all-sports-hub",
    likes: 0,
    comments: 0
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
      },
      fr: {
        title: "La Renaissance du Football Africain",
        excerpt: "Comment les pays africains réinventent la scène mondiale du football",
        content: [
          {
            type: "paragraph",
            content: "Le football africain est en train de renaître d'une manière inédite, avec plus de joueurs que jamais à marquer leur marque dans les plus hautes ligues européennes. Ce changement n'est pas seulement concerné par le talent individuel, mais représente un décalage plus large dans la manière dont les pays africains abordent le développement du football."
          },
          {
            type: "subheading",
            content: "Révolution du Développement de la Jeunesse"
          },
          {
            type: "paragraph",
            content: "Des pays comme le Sénégal et le Maroc ont investi beaucoup dans les académies de jeunesse et les infrastructures, créant des voies durables pour les talents jeunes. Les résultats sont déjà visibles, avec des équipes africaines réalisant des accomplissements historiques dans les compétitions internationales."
          },
          {
            type: "quote",
            content: "Le football africain n'est plus seulement concerné par le talent brut - c'est plutôt concerné par les systèmes de développement sophistiqués et les structures professionnelles.",
            author: "Sadio Mané"
          },
          {
            type: "paragraph",
            content: "Avec l'amélioration de l'éducation aux entraînements et les installations modernes, le continent est témoin d'un changement qui pourrait rééquilibrer la hiérarchie mondiale du football dans les décennies à venir."
          }
        ]
      },
      ar: {
        title: "تطور كرة القدم في أفريقيا: عصر جديد بدأ",
        excerpt: "كيف تعيد أندية الدوري الإفريقي تعريف الأوروبي الكرة الحديثة",
        content: [
          {
            type: "paragraph",
            content: "كرة القدم في أفريقيا تعيد الإنتاج في طريقة غير مسبقة، مع أكثر اللاعبين أن يحققوا علاماتهم في الدوريات الأوروبية العليا. هذا التطور ليس فقط عن الهوية الفردية، ولكنه يمثل تحولًا أكثر عمقًا في الطريقة التي تعتبر بها أندية الدوري الإفريقي في تطوير كرة القدم."
          },
          {
            type: "subheading",
            content: "تطور تطوير الشباب"
          },
          {
            type: "paragraph",
            content: "تم توفير الكثير من الاستثمار في المدارس الشبابية والبنية التحتية في الدوري الإفريقي، مما أنشأ مسارات مستدامة للموهوبين الشباب. النتائج بالفعل تظهر، مع أندية الدوري الإفريقي تحقق إنجازات تاريخية في المنافسات الدولية."
          },
          {
            type: "quote",
            content: "كرة القدم في أفريقيا لم تعد تتعامل مع الهوية الخام، بل تتعامل مع الأنظمة التطويرية المعقدة والهياكل المهنية.",
            author: "ساديو ماني"
          },
          {
            type: "paragraph",
            content: "مع تحسين التعليم التدريبي والمرافق الحديثة، تعتبر القارة هيئة تغيير يمكن أن يعيد تصنيف الهيراركية الموسمية في الدور القادم من العقود."
          }
        ]
      }
    },
    author: "Sedki B.Haouala",
    authorImage: sedkiImage,
    date: "January 20, 2024",
    image: article2Image,
    category: "the-beautiful-game",
    likes: 0,
    comments: 0
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
      },
      fr: {
        title: "La Révolution Numérique du Football",
        excerpt: "De la VAR à l'IA : La transformation technologique du football",
        content: [
          {
            type: "paragraph",
            content: "Le beau jeu est en train de subir une transformation digitale qui s'étend bien au-delà de la VAR. De l'analyse de performance IA à l'engagement fan blockchain, la technologie réinvente la manière dont le football est joué, regardé et expérimenté."
          },
          {
            type: "subheading",
            content: "La Révolution des Données"
          },
          {
            type: "paragraph",
            content: "Les clubs emploient désormais des scientifiques des données et des experts en apprentissage automatique pour gagner des avantages compétitifs. Chaque aspect du jeu est quantifié et analysé, de la trajectoire des joueurs à la tactique."
          },
          {
            type: "quote",
            content: "Le futur du football sera modelé par ceux qui tirent le meilleur parti de la puissance des données et de la technologie.",
            author: "Arsène Wenger"
          },
          {
            type: "paragraph",
            content: "Cette révolution technologique est démocratique en analyse du football et crée de nouvelles opportunités pour les clubs de toutes tailles de concurrencer au niveau le plus élevé."
          }
        ]
      },
      ar: {
        title: "الثورة الرقمية في كرة القدم: كيف تغيرت التكنولوجيا اللعب",
        excerpt: "من VAR إلى AI: تحول التكنولوجيا التكتيكي لكرة القدم",
        content: [
          {
            type: "paragraph",
            content: "كرة القدم تعاني من تحول رقمي يمتد خارج VAR. من تحليل الأداء الذي تم تطويره بواسطة AI إلى تواصل المعجب الذي تم تطويره بواسطة blockchain، تحول التكنولوجيا تصور كيفية اللعب والمشاهدة والتجربة لكرة القدم."
          },
          {
            type: "subheading",
            content: "الثورة البياناتية"
          },
          {
            type: "paragraph",
            content: "تستخدم الأندية الآن علماء البيانات والخبراء في التعلم الآلي لتحقيق ميزات منافسة، لأن كل جزء من اللعبة يتم تحديد الكمية والتحليل، من حركات اللاعبين إلى الأنماط التكتيكية."
          },
          {
            type: "quote",
            content: "سيصور المستقبل لكرة القدم بالذين يحصلون على أفضل جزء من قوة البيانات والتكنولوجيا.",
            author: "آرسين وينغر"
          },
          {
            type: "paragraph",
            content: "هذه الثورة التكنولوجية هي من الديموقراطية في تحليل كرة القدم وإنشاء الفرص الجديدة للأندية كلها الصغيرة لمنافسة على أعلى مستوى."
          }
        ]
      }
    },
    author: "Sedki B.Haouala",
    authorImage: sedkiImage,
    date: "January 18, 2024",
    image: article1Image,
    category: "all-sports-hub",
    likes: 0,
    comments: 0
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
      },
      fr: {
        title: "Le Jeu de Pressing Moderne : Une Analyse Profonde",
        excerpt: "Analyser comment les équipes élite ont perfectionné l'art de presser",
        content: [
          {
            type: "paragraph",
            content: "Le pressing haut est devenu une caractéristique définissante du football moderne, mais son implémentation varie de manière significative entre les équipes. Cette analyse détaille les différents styles de pressing employés par les équipes élite et leur efficacité."
          },
          {
            type: "subheading",
            content: "Les Déclencheurs de Pressing"
          },
          {
            type: "paragraph",
            content: "Comprendre quand et comment presser est crucial. Les équipes comme Manchester City et Liverpool ont développé des déclencheurs de pressing sophistiqués qui leur permettent de gagner la balle dans des positions avantageuses tout en maintenant la stabilité défensive."
          },
          {
            type: "quote",
            content: "Le pressing n'est pas concerné par le courir plus, c'est plutôt concerné par le courir plus intelligent.",
            author: "Jürgen Klopp"
          },
          {
            type: "paragraph",
            content: "La réussite d'un système de pressing dépend de la compréhension collective et du timing précis. Nous examinons comment les équipes élite coordonnent leur press et l'adaptent à différents adversaires."
          }
        ]
      },
      ar: {
        title: "لعبة الضغط الحديث: تحليل عميق تكتيكي",
        excerpt: "تحليل كيفية أندية الممتازة قد أكملت الفن الضغط في كرة القدم الحديثة",
        content: [
          {
            type: "paragraph",
            content: "ضغط عالي أصبح خاصية تحديدية لكرة القدم الحديثة، لكن تنفيذه تختلف بشكل معنوي كبير بين الأندية. تحلل هذا التحليل الأنماط المختلفة للضغط التي تم توفيرها بواسطة أندية الممتازة وفعاليتها."
          },
          {
            type: "subheading",
            content: "إثارة الضغط"
          },
          {
            type: "paragraph",
            content: "فهم متى وكيفية ضغط يعتبر عاملًا حاسمًا. تم تطوير أندية مثل مانشستر سيتي وليفربول إثارة ضغط متطورة تمكنهم من الفوز بالكرة في مواقع مفيدة في حين حماية الاستقرار الدفاعي."
          },
          {
            type: "quote",
            content: "الضغط ليس عن الجري أكثر، بل عن الجري أكثر ذكاءً.",
            author: "جورجن كلوب"
          },
          {
            type: "paragraph",
            content: "نجاز نظام الضغط يعتمد على فهم مجموعي والوقت الدقيق. نحن نستكشف كيف تنسق أندية الممتازة الضغط ويتكيف مع الأصدقاء المختلفين."
          }
        ]
      }
    },
    author: "Sedki B.Haouala",
    authorImage: sedkiImage,
    date: "January 16, 2024",
    image: article3Image,
    category: "etoile-du-sahel",
    likes: 0,
    comments: 0
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
      },
      fr: {
        title: "Évolution de la Position de Neuf Faux",
        excerpt: "Comment le rôle de l'avant-centre a évolué dans le football moderne",
        content: [
          {
            type: "paragraph",
            content: "La position de neuf fausse a évolué de manière spectaculaire de l'interprétation de Messi à des adaptations modernes par des joueurs comme Haaland. Cette analyse explore les implications tactiques et l'avenir de ce rôle révolutionnaire."
          }
        ]
      },
      ar: {
        title: "تطور موقع التسبيح",
        excerpt: "كيف تطورت دور متسبيح الأمام في كرة القدم الحديثة",
        content: [
          {
            type: "paragraph",
            content: "موقع التسبيح الخاطئ تطور بشكل عظيم من تفسير ميسي إلى تكييفات معاصرة من قبل اللاعبين مثل هالاند. تستكشف هذه التحليل التأثيرات التكتيكية والمستقبل لهذا الدور الإيجابي."
          }
        ]
      }
    },
    author: "Sedki B.Haouala",
    authorImage: sedkiImage,
    date: "January 22, 2024",
    image: article1Image,
    category: "etoile-du-sahel",
    likes: 0,
    comments: 0
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
      },
      fr: {
        title: "Analyse des Données en Recrutement de Football",
        excerpt: "Comment les données massives sont en train de révolutionner l'identification des talents",
        content: [
          {
            type: "paragraph",
            content: "Le recrutement de football moderne combine des méthodes traditionnelles avec des analyses de données sophistiquées. Nous explorons comment les clubs utilisent l'IA et l'apprentissage automatique pour découvrir la prochaine génération d'étoiles."
          }
        ]
      },
      ar: {
        title: "تحليل البيانات في البحث عن الهوية",
        excerpt: "كيف تحول البيانات الكبيرة تعريف الهوية",
        content: [
          {
            type: "paragraph",
            content: "تم توفير البحث عن الهوية في كرة القدم الحديث بجمع الطرق التقليدية مع تحليل البيانات المعقدة. نستكشف كيف تستخدم الأندية البيانات والتعلم الآلي لكشف الجيل القادم من النجوم."
          }
        ]
      }
    },
    author: "Sedki B.Haouala",
    authorImage: sedkiImage,
    date: "January 21, 2024",
    image: article2Image,
    category: "etoile-du-sahel",
    likes: 0,
    comments: 0
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
      },
      fr: {
        title: "L'Art de la Conception de Pièces de Réparation",
        excerpt: "Décryptage des routines les plus innovantes de tir et de coin",
        content: [
          {
            type: "paragraph",
            content: "Les pièces de réparation peuvent être la différence dans les matches serrés. Cette analyse examine les routines les plus créatives et les plus efficaces des équipes élite d'Europe."
          }
        ]
      },
      ar: {
        title: "فن تصميم الكرة",
        excerpt: "تحليل أكثر تطورًا لطريقة الكرة والركلة الحرة",
        content: [
          {
            type: "paragraph",
            content: "يمكن أن تكون الكرة الحرة علامة مختلفة في المباريات المضطربة. تحلل هذا التحليل أكثر الطرق الإبداعية والفعالية من الأندية الممتازة في أوروبا."
          }
        ]
      }
    },
    author: "Sedki B.Haouala",
    authorImage: sedkiImage,
    date: "January 19, 2024",
    image: article3Image,
    category: "all-sports-hub",
    likes: 0,
    comments: 0
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
      },
      fr: {
        title: "Transitions Défensives en Football Moderne",
        excerpt: "Comment les équipes élite organisent leur pression contre-",
        content: [
          {
            type: "paragraph",
            content: "Les moments immédiats après la perte de possession sont cruciaux en football moderne. Nous analysons comment les équipes élite structurent leurs transitions défensives et les mécanismes de pression contre."
          }
        ]
      },
      ar: {
        title: "تحولات الدفاع في كرة القدم الحديثة",
        excerpt: "كيف تنظم أندية الممتازة الضغط المعاكس",
        content: [
          {
            type: "paragraph",
            content: "اللحظات الفورية بعد فقدان الكرة هي عامل حاسم في كرة القدم الحديثة. نحلل كيفية بناء أندية الممتازة الدفاعية وآليات الضغط المعاكس."
          }
        ]
      }
    },
    author: "Sedki B.Haouala",
    authorImage: sedkiImage,
    date: "January 17, 2024",
    image: article1Image,
    category: "etoile-du-sahel",
    likes: 0,
    comments: 0
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
      },
      fr: {
        title: "Évolution des Gardiens de But",
        excerpt: "Analyser le rôle élargi des gardiens de but dans le jeu de base",
        content: [
          {
            type: "paragraph",
            content: "Les gardiens de but modernes sont attendus pour être compétents avec leurs pieds et participer au jeu de base. Nous examinons comment cette évolution a changé les tactiques de football."
          }
        ]
      },
      ar: {
        title: "تطور مرمي الهدف",
        excerpt: "تحليل دور مرمي الهدف في اللعب الأساسي",
        content: [
          {
            type: "paragraph",
            content: "مرمي الهدف الحديث يتوقع أن يكون مهاراً مع الأقدام ويشارك في اللعب الأساسي. نحلل كيف تطور هذا التطور على التكتيك."
          }
        ]
      }
    },
    author: "Sedki B.Haouala",
    authorImage: sedkiImage,
      date: "January 15, 2024",
    image: article2Image,
    category: "etoile-du-sahel",
    likes: 0,
    comments: 0
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
      },
      fr: {
        title: "De Réfugié à Champion : Le Voyage Inspirant de Mohamed Ali",
        excerpt: "Histoire de persévérance, d'espoir et de football comme langage universel",
        content: [
          {
            type: "paragraph",
            content: "Le voyage de Mohamed Ali, d'un camp de réfugié à football professionnel, est un témoignage de la puissance des rêves et de la détermination."
          }
        ]
      },
      ar: {
        title: "من مهاجر إلى بطل: رحلة تثرية لموهب موحد",
        excerpt: "قصة تحمل وطنية وطنية وكرة القدم كلام عالمي",
        content: [
          {
            type: "paragraph",
            content: "رحلة موهب موحد من مهاجر إلى كرة القدم المهنية هي شهادة لقوة الأحلام والتحديد."
          }
        ]
      }
    },
    author: "Sedki B.Haouala",
    authorImage: sedkiImage,
    date: "January 21, 2024",
    image: article3Image,
    category: "the-beautiful-game",
    likes: 0,
    comments: 0
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
      },
      fr: {
        title: "La Révolution de la Ligue Dimanche",
        excerpt: "Comment le football de rue est en train d'embrasser la technologie et la professionnalisation",
        content: [
          {
            type: "paragraph",
            content: "Le football amateur est en train de subir une transformation avec des applications, la technologie de suivi et les méthodes d'entraînement professionnel de plus en plus accessibles."
          }
        ]
      },
      ar: {
        title: "الدوري الأسبوعي الإيجابي",
        excerpt: "كيف تعترف الكرة المدنية مع التكنولوجيا والمهنية",
        content: [
          {
            type: "paragraph",
            content: "كرة القدم المدنية تعاني تطورًا مع ظهور تطبيقات وتكنولوجيا المتابعة وطرق التدريب المهنية الأكثر توافرًا."
          }
        ]
      }
    },
    author: "Sedki B.Haouala",
    authorImage: sedkiImage,
    date: "January 20, 2024",
    image: article1Image,
    category: "the-beautiful-game",
    likes: 0,
    comments: 0
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
      },
      fr: {
        title: "La Révolution de la Santé Mentale du Football",
        excerpt: "Comment les clubs priorisent la santé mentale des joueurs au-delà du terrain",
        content: [
          {
            type: "paragraph",
            content: "Le football professionnel est enfin abordant la santé mentale ouvertement, avec les clubs mettant en œuvre des systèmes de soutien complets pour leurs joueurs."
          }
        ]
      },
      ar: {
        title: "الثورة النفسية لكرة القدم",
        excerpt: "كيف تعترف الأندية بصحة اللاعبين خارج الملعب",
        content: [
          {
            type: "paragraph",
            content: "كرة القدم المهنية أخيراً تواجه صحة اللاعبين علنيًا، مع توفير أنظمة الدعم الشاملة لللاعبين."
          }
        ]
      }
    },
    author: "Sedki B.Haouala",
    authorImage: sedkiImage,
    date: "January 19, 2024",
    image: article2Image,
    category: "the-beautiful-game",
    likes: 0,
    comments: 0
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
      },
      fr: {
        title: "Le Dernier Club Amateur",
        excerpt: "Comment un club historique maintient son statut amateur dans la période professionnelle",
        content: [
          {
            type: "paragraph",
            content: "Dans un âge d'industrialisation, un club historique continue de maintenir les valeurs amateurs tout en concourant contre des équipes professionnelles."
          }
        ]
      },
      ar: {
        title: "أخير نادي مدني",
        excerpt: "كيف تحتفظ نادي تاريخي بحالته المدنية في العصر المهني",
        content: [
          {
            type: "paragraph",
            content: "في عصر التجارة، يحتفظ نادي تاريخي بالقيم المدنية في حين يتنافس على الأندية المهنية."
          }
        ]
      }
    },
    author: "Sedki B.Haouala",
    authorImage: sedkiImage,
    date: "January 18, 2024",
    image: article3Image,
    category: "the-beautiful-game",
    likes: 0,
    comments: 0
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
      },
      fr: {
        title: "Le Défi Climatique du Football",
        excerpt: "Comment les clubs s'adaptent aux préoccupations environnementales",
        content: [
          {
            type: "paragraph",
            content: "Des stades solaires aux voyages à neutralité carbone, les clubs de football prennent des mesures innovantes pour répondre au défi climatique."
          }
        ]
      },
      ar: {
        title: "تحدي كرة القدم للبيئة",
        excerpt: "كيف تعترف الأندية بالمخاوف البيئية",
        content: [
          {
            type: "paragraph",
            content: "من الملاعب الشمسية إلى السفر بالتعادل الكربوني، تتبنى أندية كرة القدم خطوات تكنولوجية للتعامل مع تحدي البيئة."
          }
        ]
      }
    },
    author: "Sedki B.Haouala",
    authorImage: sedkiImage,
    date: "January 17, 2024",
    image: article1Image,
    category: "all-sports-hub",
    likes: 0,
    comments: 0
  }
].map(article => ({
  ...article,
  author: 'Sedki B.Haouala',
  authorImage: sedkiImage
}));

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