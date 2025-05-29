const express = require('express');
const { check } = require('express-validator');
const newsletterController = require('../controllers/newsletterController');
const auth = require('../middleware/auth');
const isAdmin = require('../middleware/isAdmin');

const router = express.Router();

// @route   POST /api/newsletter/subscribe
// @desc    Subscribe to newsletter
// @access  Public
router.post('/subscribe', [
    check('email', 'Please include a valid email').isEmail(),
    check('preferences').optional().isObject()
], newsletterController.subscribe);

// @route   GET /api/newsletter/verify/:token
// @desc    Verify newsletter subscription
// @access  Public
router.get('/verify/:token', newsletterController.verifySubscription);

// @route   GET /api/newsletter/unsubscribe/:token
// @desc    Unsubscribe from newsletter
// @access  Public
router.get('/unsubscribe/:token', newsletterController.unsubscribe);

// @route   PUT /api/newsletter/preferences/:token
// @desc    Update subscription preferences
// @access  Public
router.put('/preferences/:token', [
    check('preferences').isObject()
], newsletterController.updatePreferences);

// Admin routes
// @route   POST /api/newsletter
// @desc    Create newsletter
// @access  Private/Admin
router.post('/', [
    auth,
    isAdmin,
    [
        check('subject', 'Subject is required').not().isEmpty(),
        check('content', 'Content is required').not().isEmpty(),
        check('category', 'Category is required').isIn(['weekly-digest', 'breaking-news', 'feature-article'])
    ]
], newsletterController.createNewsletter);

// @route   POST /api/newsletter/:id/send
// @desc    Send newsletter
// @access  Private/Admin
router.post('/:id/send', [auth, isAdmin], newsletterController.sendNewsletter);

module.exports = router; 
 