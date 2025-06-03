const mongoose = require('mongoose');
const slugify = require('slugify');

const contentBlockSchema = new mongoose.Schema({
    type: {
        type: String,
        required: true,
        enum: ['paragraph', 'heading', 'quote', 'image', 'image-group', 'list']
    },
    content: {
        type: String,
        required: true
    },
    metadata: {
        level: Number, // For headings (h2, h3, etc.)
        source: String, // For quotes
        caption: String, // For images
        alignment: {
            type: String,
            enum: ['left', 'center', 'right', 'justify']
        },
        style: {
            margins: {
                top: Number,
                bottom: Number
            },
            textColor: String,
            backgroundColor: String
        },
        listType: {
            type: String,
            enum: ['bullet', 'numbered']
        },
        images: [{
            url: String,
            caption: String,
            alignment: String,
            width: Number,
            height: Number
        }]
    }
});

const translationSchema = new mongoose.Schema({
    title: {
        type: String,
        required: true
    },
    excerpt: {
        type: String,
        required: true
    },
    content: {
        type: [contentBlockSchema],
        required: true,
        default: []
    },
    // For backward compatibility
    legacyContent: {
        type: String
    }
});

const articleSchema = new mongoose.Schema({
    translations: {
        en: {
            type: translationSchema,
            required: true
        },
        ar: {
            type: translationSchema,
            required: true
        }
    },
    author: {
        type: String,
        required: true
    },
    authorImage: {
        type: String,
        required: true
    },
    date: {
        type: Date,
        default: Date.now
    },
    image: {
        type: String,
        required: true
    },
    category: {
        type: String,
        enum: ['etoile-du-sahel', 'the-beautiful-game', 'all-sports-hub'],
        required: true
    },
    status: {
        type: String,
        enum: ['draft', 'published', 'archived'],
        default: 'published'
    },
    publishedAt: {
        type: Date
    },
    views: {
        type: Number,
        default: 0
    },
    likes: {
        count: {
            type: Number,
            default: 0
        },
        users: [{
            type: mongoose.Schema.Types.ObjectId,
            ref: 'User'
        }]
    },
    shares: {
        count: {
            type: Number,
            default: 0
        },
        platforms: {
            twitter: { type: Number, default: 0 },
            facebook: { type: Number, default: 0 },
            linkedin: { type: Number, default: 0 }
        }
    },
    slug: {
        type: String,
        unique: true
    }
}, {
    timestamps: true,
    toJSON: { virtuals: true },
    toObject: { virtuals: true }
});

// Add text index for search functionality
articleSchema.index({ 
    'translations.en.title': 'text',
    'translations.ar.title': 'text',
    'translations.en.excerpt': 'text',
    'translations.ar.excerpt': 'text',
    'translations.en.content': 'text',
    'translations.ar.content': 'text'
});

// Virtual for comments
articleSchema.virtual('commentCount', {
    ref: 'Comment',
    localField: '_id',
    foreignField: 'article',
    count: true
});

// Check for duplicate title before saving
articleSchema.pre('save', async function(next) {
    if (this.isModified('translations.en.title')) {
        const existingArticle = await this.constructor.findOne({
            'translations.en.title': this.translations.en.title,
            _id: { $ne: this._id } // Exclude current article when updating
        });

        if (existingArticle) {
            next(new Error('An article with this title already exists. Please choose a different title.'));
            return;
        }

        this.slug = slugify(this.translations.en.title, {
            lower: true,
            strict: true
        });
    }
    
    if (this.status === 'published' && !this.publishedAt) {
        this.publishedAt = new Date();
    }
    
    next();
});

// Method to increment view count
articleSchema.methods.incrementViews = async function() {
    this.views += 1;
    return this.save();
};

// Method to handle likes
articleSchema.methods.toggleLike = async function(userId) {
    const userIndex = this.likes.users.indexOf(userId);
    
    if (userIndex === -1) {
        this.likes.users.push(userId);
        this.likes.count += 1;
    } else {
        this.likes.users.splice(userIndex, 1);
        this.likes.count -= 1;
    }
    
    return this.save();
};

// Method to increment share count
articleSchema.methods.incrementShare = async function(platform) {
    if (this.shares.platforms[platform] !== undefined) {
        this.shares.platforms[platform] += 1;
        this.shares.count += 1;
        return this.save();
    }
};

module.exports = mongoose.model('Article', articleSchema); 
 