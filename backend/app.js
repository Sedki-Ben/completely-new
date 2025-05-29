const express = require('express');
const cors = require('cors');
const { i18nextMiddleware } = require('./utils/i18n');
const { limiter, authLimiter, newsletterLimiter } = require('./middleware/rateLimiter');

const app = express();

// CORS Configuration
const corsOptions = {
  origin: process.env.FRONTEND_URL || 'http://localhost:3000',
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'Accept-Language']
};

app.use(cors(corsOptions));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Apply i18n middleware before routes
app.use(i18nextMiddleware);

// Apply rate limiting to all routes
app.use(limiter);

// Apply specific rate limits to auth routes
app.use('/api/auth/login', authLimiter);
app.use('/api/auth/register', authLimiter);

// Apply rate limit to newsletter subscription
app.use('/api/newsletter/subscribe', newsletterLimiter);

// Import routes
const authRoutes = require('./routes/auth');
const articleRoutes = require('./routes/articles');
const newsletterRoutes = require('./routes/newsletter');
const userRoutes = require('./routes/users');

// Use routes
app.use('/api/auth', authRoutes);
app.use('/api/articles', articleRoutes);
app.use('/api/newsletter', newsletterRoutes);
app.use('/api/users', userRoutes);

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  const status = err.status || 500;
  const message = err.message || 'Something went wrong!';
  
  res.status(status).json({
    status: 'error',
    message,
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack })
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    status: 'error',
    message: 'Route not found'
  });
});

module.exports = app; 
 