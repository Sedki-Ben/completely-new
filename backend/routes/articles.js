const express = require('express');
const { check } = require('express-validator');
const articleController = require('../controllers/articleController');
const auth = require('../middleware/auth');

const router = express.Router();

// @route   POST /api/articles
// @desc    Create a new article
// @access  Private
router.post('/', [
    auth,
    [
        check('title', 'Title is required').not().isEmpty(),
        check('content', 'Content is required').not().isEmpty(),
        check('summary', 'Summary is required').not().isEmpty(),
        check('category', 'Category is required').isIn(['analysis', 'stories', 'notable-work']),
        check('language', 'Language is required').isIn(['en', 'fr', 'ar'])
    ]
], articleController.createArticle);

// @route   GET /api/articles
// @desc    Get all articles with filters
// @access  Public
router.get('/', articleController.getArticles);

// @route   GET /api/articles/search
// @desc    Search articles
// @access  Public
router.get('/search', articleController.searchArticles);

// @route   GET /api/articles/:id
// @desc    Get single article
// @access  Public
router.get('/:id', articleController.getArticle);

// @route   PUT /api/articles/:id
// @desc    Update article
// @access  Private
router.put('/:id', [
    auth,
    [
        check('title', 'Title is required').optional().not().isEmpty(),
        check('content', 'Content is required').optional().not().isEmpty(),
        check('summary', 'Summary is required').optional().not().isEmpty(),
        check('category', 'Invalid category').optional().isIn(['analysis', 'stories', 'notable-work']),
        check('language', 'Invalid language').optional().isIn(['en', 'fr', 'ar'])
    ]
], articleController.updateArticle);

// @route   DELETE /api/articles/:id
// @desc    Delete article
// @access  Private
router.delete('/:id', auth, articleController.deleteArticle);

// @route   POST /api/articles/:id/like
// @desc    Like/Unlike article
// @access  Private
router.post('/:id/like', auth, articleController.toggleLike);

module.exports = router; 
 