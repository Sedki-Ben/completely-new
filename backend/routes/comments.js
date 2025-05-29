const express = require('express');
const { check } = require('express-validator');
const commentController = require('../controllers/commentController');
const auth = require('../middleware/auth');

const router = express.Router();

// @route   POST /api/articles/:articleId/comments
// @desc    Create a comment
// @access  Private
router.post('/:articleId/comments', [
    auth,
    [
        check('content', 'Content is required').not().isEmpty(),
        check('parentId', 'Parent comment ID must be valid if provided').optional().isMongoId()
    ]
], commentController.createComment);

// @route   GET /api/articles/:articleId/comments
// @desc    Get comments for an article
// @access  Public
router.get('/:articleId/comments', commentController.getComments);

// @route   PUT /api/articles/:articleId/comments/:commentId
// @desc    Update a comment
// @access  Private
router.put('/:articleId/comments/:commentId', [
    auth,
    [
        check('content', 'Content is required').not().isEmpty()
    ]
], commentController.updateComment);

// @route   DELETE /api/articles/:articleId/comments/:commentId
// @desc    Delete a comment
// @access  Private
router.delete('/:articleId/comments/:commentId', auth, commentController.deleteComment);

// @route   POST /api/articles/:articleId/comments/:commentId/like
// @desc    Like/Unlike a comment
// @access  Private
router.post('/:articleId/comments/:commentId/like', auth, commentController.toggleLike);

module.exports = router; 
 