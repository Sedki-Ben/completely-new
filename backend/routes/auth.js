const express = require('express');
const { check } = require('express-validator');
const authController = require('../controllers/authController');
const auth = require('../middleware/auth');
const isAdmin = require('../middleware/isAdmin');

const router = express.Router();

// @route   POST /api/auth/register
// @desc    Register user
// @access  Public
router.post('/register', [
    check('name', 'Name is required').not().isEmpty(),
    check('email', 'Please include a valid email').isEmail(),
    check('password', 'Please enter a password with 6 or more characters').isLength({ min: 6 })
], authController.register);

// @route   POST /api/auth/login
// @desc    Login user
// @access  Public
router.post('/login', [
    check('email', 'Please include a valid email').isEmail(),
    check('password', 'Password is required').exists()
], authController.login);

// @route   POST /api/auth/logout
// @desc    Logout user
// @access  Private
router.post('/logout', auth, authController.logout);

// @route   GET /api/auth/me
// @desc    Get current user
// @access  Private
router.get('/me', auth, authController.getCurrentUser);

// @route   PUT /api/auth/profile
// @desc    Update user profile
// @access  Private
router.put('/profile', [
    auth,
    check('name', 'Name is required').optional().not().isEmpty(),
    check('bio').optional().isLength({ max: 500 }),
    check('profilePicture').optional().isURL()
], authController.updateProfile);

// @route   PUT /api/auth/password
// @desc    Change password
// @access  Private
router.put('/password', [
    auth,
    check('currentPassword', 'Current password is required').exists(),
    check('newPassword', 'Please enter a new password with 6 or more characters').isLength({ min: 6 })
], authController.changePassword);

// @route   POST /api/auth/forgot-password
// @desc    Request password reset
// @access  Public
router.post('/forgot-password', [
    check('email', 'Please include a valid email').isEmail()
], authController.forgotPassword);

// @route   POST /api/auth/reset-password/:token
// @desc    Reset password
// @access  Public
router.post('/reset-password/:token', [
    check('password', 'Please enter a password with 6 or more characters').isLength({ min: 6 })
], authController.resetPassword);

// @route   PUT /api/auth/language
// @desc    Update user language
// @access  Private
router.put('/language', auth, authController.updateLanguage);

// Admin Routes
// @route   GET /api/auth/users
// @desc    Get all users (admin only)
// @access  Private/Admin
router.get('/users', [auth, isAdmin], authController.getAllUsers);

// @route   PUT /api/auth/users/:id/role
// @desc    Update user role (admin only)
// @access  Private/Admin
router.put('/users/:id/role', [
    auth,
    isAdmin,
    check('role').isIn(['user', 'writer', 'admin'])
], authController.updateUserRole);

// @route   DELETE /api/auth/users/:id
// @desc    Delete user (admin only)
// @access  Private/Admin
router.delete('/users/:id', [auth, isAdmin], authController.deleteUser);

module.exports = router; 
 