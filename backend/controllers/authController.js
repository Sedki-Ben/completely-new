const jwt = require('jsonwebtoken');
const { validationResult } = require('express-validator');
const User = require('../models/User');

// Generate JWT Token
const generateToken = (userId) => {
    return jwt.sign({ userId }, process.env.JWT_SECRET || 'your_jwt_secret_key_here', {
        expiresIn: '24h'
    });
};

// Register new user
exports.register = async (req, res) => {
    try {
        // Validate request
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res.status(400).json({ errors: errors.array() });
        }

        const { name, email, password, language } = req.body;

        // Check if user already exists
        let user = await User.findOne({ email });
        if (user) {
            return res.status(400).json({ message: 'User already exists' });
        }

        // Create new user
        const userData = { name, email, password };
        if (language) userData.language = language;
        user = new User(userData);

        await user.save();

        // Generate token
        const token = generateToken(user._id);

        res.status(201).json({
            token,
            user: {
                id: user._id,
                name: user.name,
                email: user.email,
                language: user.language
            }
        });
    } catch (error) {
        console.error('Register error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// Login user
exports.login = async (req, res) => {
    try {
        const { email, password } = req.body;

        // Find user
        const user = await User.findOne({ email });
        if (!user) {
            return res.status(401).json({ message: 'Invalid credentials' });
        }

        // Check password
        const isMatch = await user.comparePassword(password);
        if (!isMatch) {
            return res.status(401).json({ message: 'Invalid credentials' });
        }

        // Generate token
        const token = generateToken(user._id);

        res.json({
            token,
            user: {
                id: user._id,
                name: user.name,
                email: user.email,
                role: user.role,
                language: user.language
            }
        });
    } catch (error) {
        console.error('Login error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// Get current user
exports.getCurrentUser = async (req, res) => {
    try {
        const user = await User.findById(req.user.userId).select('-password');
        res.json(user);
    } catch (error) {
        console.error('Get current user error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// Update user language
exports.updateLanguage = async (req, res) => {
    try {
        const { language } = req.body;
        if (!['en', 'fr', 'ar'].includes(language)) {
            return res.status(400).json({ message: 'Invalid language' });
        }

        const user = await User.findByIdAndUpdate(
            req.user.userId,
            { language },
            { new: true }
        ).select('-password');

        res.json(user);
    } catch (error) {
        console.error('Update language error:', error);
        res.status(500).json({ message: 'Server error' });
    }
};

// Logout user
exports.logout = (req, res) => {
    res.json({ message: 'Logged out' });
};

// Update user profile
exports.updateProfile = (req, res) => {
    res.json({ message: 'Profile updated (stub)' });
};

// Change password
exports.changePassword = (req, res) => {
    res.json({ message: 'Password changed (stub)' });
};

// Forgot password
exports.forgotPassword = (req, res) => {
    res.json({ message: 'Password reset email sent (stub)' });
};

// Reset password
exports.resetPassword = (req, res) => {
    res.json({ message: 'Password reset (stub)' });
};

// Get all users (admin)
exports.getAllUsers = (req, res) => {
    res.json({ users: [] });
};

// Update user role (admin)
exports.updateUserRole = (req, res) => {
    res.json({ message: 'User role updated (stub)' });
};

// Delete user (admin)
exports.deleteUser = (req, res) => {
    res.json({ message: 'User deleted (stub)' });
}; 
 