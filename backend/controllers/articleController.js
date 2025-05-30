const Article = require('../models/Article');
const { validationResult } = require('express-validator');
const { Subscription } = require('../models/Newsletter');
const EmailService = require('../utils/emailService');

// Create article
exports.createArticle = async (req, res) => {
    try {
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res.status(400).json({ errors: errors.array() });
        }

        const articleData = {
            ...req.body,
            author: req.user.userId
        };
        if (req.file) {
            articleData.coverImage = `/uploads/${req.file.filename}`;
        }

        const article = new Article(articleData);
        await article.save();

        // Notify newsletter subscribers if article is published
        if (article.status === 'published') {
            try {
                const subscribers = await Subscription.find({ isVerified: true });
                if (subscribers.length > 0) {
                    await EmailService.sendArticleNotification(subscribers, {
                        title: article.title,
                        summary: article.excerpt || '',
                        _id: article._id
                    });
                }
            } catch (notifyErr) {
                console.error('Error sending article notification:', notifyErr);
            }
        }

        res.status(201).json(article);
    } catch (error) {
        console.error('Create article error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// Get all articles with filters
exports.getArticles = async (req, res) => {
    try {
        const { category, language, status, page = 1, limit = 10 } = req.query;
        const query = {};

        if (category) query.category = category;
        if (language) query.language = language;
        if (status) query.status = status;

        const articles = await Article.find(query)
            .populate('author', 'name')
            .sort({ createdAt: -1 })
            .limit(limit * 1)
            .skip((page - 1) * limit)
            .exec();

        const count = await Article.countDocuments(query);

        res.json({
            articles,
            totalPages: Math.ceil(count / limit),
            currentPage: page
        });
    } catch (error) {
        console.error('Get articles error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// Get single article
exports.getArticle = async (req, res) => {
    try {
        const article = await Article.findById(req.params.id)
            .populate('author', 'name')
            .populate({
                path: 'commentCount'
            });

        if (!article) {
            return res.status(404).json({ message: 'Article not found' });
        }

        // Increment views
        article.views += 1;
        await article.save();

        res.json(article);
    } catch (error) {
        console.error('Get article error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// Update article
exports.updateArticle = async (req, res) => {
    try {
        const article = await Article.findById(req.params.id);

        if (!article) {
            return res.status(404).json({ message: 'Article not found' });
        }

        // Check ownership
        if (article.author.toString() !== req.user.userId) {
            return res.status(403).json({ message: 'Not authorized' });
        }

        const updateData = { ...req.body };
        if (req.file) {
            updateData.coverImage = `/uploads/${req.file.filename}`;
        }

        const updatedArticle = await Article.findByIdAndUpdate(
            req.params.id,
            updateData,
            { new: true }
        );

        res.json(updatedArticle);
    } catch (error) {
        console.error('Update article error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// Delete article
exports.deleteArticle = async (req, res) => {
    try {
        const article = await Article.findById(req.params.id);

        if (!article) {
            return res.status(404).json({ message: 'Article not found' });
        }

        // Check ownership
        if (article.author.toString() !== req.user.userId) {
            return res.status(403).json({ message: 'Not authorized' });
        }

        await article.remove();
        res.json({ message: 'Article removed' });
    } catch (error) {
        console.error('Delete article error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// Like/Unlike article
exports.toggleLike = async (req, res) => {
    try {
        const article = await Article.findById(req.params.id);

        if (!article) {
            return res.status(404).json({ message: 'Article not found' });
        }

        const likeIndex = article.likes.indexOf(req.user.userId);

        if (likeIndex > -1) {
            // Unlike
            article.likes.splice(likeIndex, 1);
        } else {
            // Like
            article.likes.push(req.user.userId);
        }

        await article.save();
        res.json({ likes: article.likes.length });
    } catch (error) {
        console.error('Toggle like error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// Search articles
exports.searchArticles = async (req, res) => {
    try {
        const { q, page = 1, limit = 10 } = req.query;

        const articles = await Article.find(
            { $text: { $search: q } },
            { score: { $meta: 'textScore' } }
        )
            .sort({ score: { $meta: 'textScore' } })
            .limit(limit * 1)
            .skip((page - 1) * limit)
            .populate('author', 'name')
            .exec();

        const count = await Article.countDocuments({ $text: { $search: q } });

        res.json({
            articles,
            totalPages: Math.ceil(count / limit),
            currentPage: page
        });
    } catch (error) {
        console.error('Search articles error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// Get articles by type (analysis, story, notable)
exports.getArticlesByType = (req, res) => {
    res.json({ articles: [] });
};

// Get article by slug
exports.getArticleBySlug = (req, res) => {
    res.json({ article: null });
};

// Record article share
exports.recordShare = (req, res) => {
    res.json({ message: 'Share recorded (stub)' });
};

// Get users who liked the article
exports.getArticleLikes = (req, res) => {
    res.json({ likes: [] });
};

// Get stats for writer's articles
exports.getWriterStats = (req, res) => {
    res.json({ stats: {} });
};

// Get writer's draft articles
exports.getWriterDrafts = (req, res) => {
    res.json({ drafts: [] });
};

// Publish a draft article
exports.publishArticle = (req, res) => {
    res.json({ message: 'Article published (stub)' });
};

// Archive an article
exports.archiveArticle = (req, res) => {
    res.json({ message: 'Article archived (stub)' });
}; 
 