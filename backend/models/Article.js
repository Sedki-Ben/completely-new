const mongoose = require('mongoose');
const slugify = require('slugify');

const articleSchema = new mongoose.Schema({
    title: {
        type: String,
        required: true,
        trim: true
    },
    content: {
        type: String,
        required: true
    },
    type: {
        type: String,
        enum: ['etoile-du-sahel', 'the-beautiful-game', 'all-sports-hub'],
        required: true
    },
    slug: {
        type: String,
        unique: true
    },
    author: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    images: [{
        url: {
            type: String,
            required: true
        },
        position: {
            type: Number,
            default: 0
        },
        caption: {
        type: String
        }
    }],
    excerpt: {
        type: String,
        maxLength: 300
    },
    tags: [{
        type: String,
        trim: true
    }],
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
    views: {
        type: Number,
        default: 0
    },
    status: {
        type: String,
        enum: ['draft', 'published', 'archived'],
        default: 'published'
    },
    publishedAt: {
        type: Date
    }
}, {
    timestamps: true,
    toJSON: { virtuals: true },
    toObject: { virtuals: true }
});

// Add text index for search functionality
articleSchema.index({ title: 'text', content: 'text', excerpt: 'text' });

// Virtual for comments
articleSchema.virtual('comments', {
    ref: 'Comment',
    localField: '_id',
    foreignField: 'article'
});

// Generate slug before saving
articleSchema.pre('save', function(next) {
    if (this.isModified('title')) {
        this.slug = slugify(this.title, {
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

const Article = mongoose.model('Article', articleSchema);

module.exports = Article; 
 