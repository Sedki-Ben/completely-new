const mongoose = require('mongoose');

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
    summary: {
        type: String,
        required: true,
        maxLength: 200
    },
    author: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    category: {
        type: String,
        required: true,
        enum: ['analysis', 'stories', 'notable-work']
    },
    tags: [{
        type: String,
        trim: true
    }],
    image: {
        url: String,
        alt: String
    },
    likes: [{
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User'
    }],
    views: {
        type: Number,
        default: 0
    },
    language: {
        type: String,
        enum: ['en', 'fr', 'ar'],
        required: true
    },
    status: {
        type: String,
        enum: ['draft', 'published', 'archived'],
        default: 'draft'
    },
    publishedAt: {
        type: Date
    }
}, {
    timestamps: true
});

// Add text index for search functionality
articleSchema.index({ title: 'text', content: 'text', summary: 'text' });

// Virtual for comment count
articleSchema.virtual('commentCount', {
    ref: 'Comment',
    localField: '_id',
    foreignField: 'article',
    count: true
});

const Article = mongoose.model('Article', articleSchema);

module.exports = Article; 
 