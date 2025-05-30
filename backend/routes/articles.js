const express = require('express');
const { check } = require('express-validator');
const articlesController = require('../controllers/articleController');
const auth = require('../middleware/auth');
const isWriter = require('../middleware/isWriter');
const multer = require('multer');
const path = require('path');

const router = express.Router();

// Multer setup for image uploads
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, path.join(__dirname, '../uploads/'));
    },
    filename: function (req, file, cb) {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, uniqueSuffix + '-' + file.originalname);
    }
});
const upload = multer({ storage });

// @route   GET /api/articles
// @desc    Get all articles with filters
// @access  Public
router.get('/', articlesController.getArticles);

// @route   GET /api/articles/:type
// @desc    Get articles by type (analysis, story, notable)
// @access  Public
router.get('/type/:type', [
    check('type').isIn(['analysis', 'story', 'notable'])
], articlesController.getArticlesByType);

// @route   GET /api/articles/slug/:slug
// @desc    Get article by slug
// @access  Public
router.get('/slug/:slug', articlesController.getArticleBySlug);

// @route   POST /api/articles
// @desc    Create new article
// @access  Private/Writer
router.post('/',
    auth,
    isWriter,
    upload.single('image'),
    [
        check('title', 'Title is required').not().isEmpty(),
        check('content', 'Content is required').not().isEmpty(),
        check('type').isIn(['analysis', 'story', 'notable']),
        check('tags').optional().isArray(),
        check('status').optional().isIn(['draft', 'published', 'archived'])
    ],
    articlesController.createArticle
);

// @route   PUT /api/articles/:id
// @desc    Update article
// @access  Private/Writer
router.put('/:id',
    auth,
    isWriter,
    upload.single('image'),
    [
        check('title').optional().not().isEmpty(),
        check('content').optional().not().isEmpty(),
        check('type').optional().isIn(['analysis', 'story', 'notable']),
        check('tags').optional().isArray(),
        check('status').optional().isIn(['draft', 'published', 'archived'])
    ],
    articlesController.updateArticle
);

// @route   DELETE /api/articles/:id
// @desc    Delete article
// @access  Private/Writer
router.delete('/:id', [auth, isWriter], articlesController.deleteArticle);

// Interaction Routes

// @route   POST /api/articles/:id/like
// @desc    Toggle like on article
// @access  Private
router.post('/:id/like', auth, articlesController.toggleLike);

// @route   POST /api/articles/:id/share
// @desc    Record article share
// @access  Public
router.post('/:id/share', [
    check('platform').isIn(['twitter', 'facebook', 'linkedin'])
], articlesController.recordShare);

// @route   GET /api/articles/:id/likes
// @desc    Get users who liked the article
// @access  Public
router.get('/:id/likes', articlesController.getArticleLikes);

// Writer/Admin Routes

// @route   GET /api/articles/stats/me
// @desc    Get stats for writer's articles
// @access  Private/Writer
router.get('/stats/me', [auth, isWriter], articlesController.getWriterStats);

// @route   GET /api/articles/drafts/me
// @desc    Get writer's draft articles
// @access  Private/Writer
router.get('/drafts/me', [auth, isWriter], articlesController.getWriterDrafts);

// @route   POST /api/articles/:id/publish
// @desc    Publish a draft article
// @access  Private/Writer
router.post('/:id/publish', [auth, isWriter], articlesController.publishArticle);

// @route   POST /api/articles/:id/archive
// @desc    Archive an article
// @access  Private/Writer
router.post('/:id/archive', [auth, isWriter], articlesController.archiveArticle);

module.exports = router; 
 