const Comment = require('../models/Comment');
const Article = require('../models/Article');
const { validationResult } = require('express-validator');

// Create comment
exports.createComment = async (req, res) => {
    try {
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res.status(400).json({ errors: errors.array() });
        }

        const article = await Article.findById(req.params.articleId);
        if (!article) {
            return res.status(404).json({ message: 'Article not found' });
        }

        const comment = new Comment({
            content: req.body.content,
            author: req.user.userId,
            article: req.params.articleId,
            parent: req.body.parentId // Optional, for replies
        });

        await comment.save();

        const populatedComment = await Comment.findById(comment._id)
            .populate('author', 'name')
            .populate('parent');

        res.status(201).json(populatedComment);
    } catch (error) {
        console.error('Create comment error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// Get comments for article
exports.getComments = async (req, res) => {
    try {
        const { page = 1, limit = 10 } = req.query;

        // Get top-level comments first (no parent)
        const comments = await Comment.find({
            article: req.params.articleId,
            parent: null
        })
            .populate('author', 'name')
            .populate({
                path: 'replies',
                populate: {
                    path: 'author',
                    select: 'name'
                }
            })
            .sort({ createdAt: -1 })
            .limit(limit * 1)
            .skip((page - 1) * limit)
            .exec();

        const count = await Comment.countDocuments({
            article: req.params.articleId,
            parent: null
        });

        res.json({
            comments,
            totalPages: Math.ceil(count / limit),
            currentPage: page
        });
    } catch (error) {
        console.error('Get comments error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// Update comment
exports.updateComment = async (req, res) => {
    try {
        const comment = await Comment.findById(req.params.commentId);

        if (!comment) {
            return res.status(404).json({ message: 'Comment not found' });
        }

        // Check ownership
        if (comment.author.toString() !== req.user.userId) {
            return res.status(403).json({ message: 'Not authorized' });
        }

        comment.content = req.body.content;
        comment.isEdited = true;
        await comment.save();

        const updatedComment = await Comment.findById(comment._id)
            .populate('author', 'name')
            .populate('parent');

        res.json(updatedComment);
    } catch (error) {
        console.error('Update comment error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// Delete comment
exports.deleteComment = async (req, res) => {
    try {
        const comment = await Comment.findById(req.params.commentId);

        if (!comment) {
            return res.status(404).json({ message: 'Comment not found' });
        }

        // Check ownership
        if (comment.author.toString() !== req.user.userId) {
            return res.status(403).json({ message: 'Not authorized' });
        }

        await comment.remove();
        res.json({ message: 'Comment removed' });
    } catch (error) {
        console.error('Delete comment error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// Toggle like on comment
exports.toggleLike = async (req, res) => {
    try {
        const comment = await Comment.findById(req.params.commentId);

        if (!comment) {
            return res.status(404).json({ message: 'Comment not found' });
        }

        const likeIndex = comment.likes.indexOf(req.user.userId);

        if (likeIndex > -1) {
            // Unlike
            comment.likes.splice(likeIndex, 1);
        } else {
            // Like
            comment.likes.push(req.user.userId);
        }

        await comment.save();
        res.json({ likes: comment.likes.length });
    } catch (error) {
        console.error('Toggle comment like error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// Get all comments for an article
exports.getArticleComments = (req, res) => {
    res.json({ comments: [] });
};

// Get replies to a comment
exports.getCommentReplies = (req, res) => {
    res.json({ replies: [] });
};

// Get users who liked the comment
exports.getCommentLikes = (req, res) => {
    res.json({ likes: [] });
};

// Report a comment
exports.reportComment = (req, res) => {
    res.json({ message: 'Comment reported (stub)' });
}; 
 