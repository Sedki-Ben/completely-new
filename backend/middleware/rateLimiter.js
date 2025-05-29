const rateLimit = require('express-rate-limit');
const RedisStore = require('rate-limit-redis');
const redisService = require('../utils/redisService');

// Create a limiter with Redis store
const limiter = rateLimit({
    store: new RedisStore({
        client: redisService.client,
        prefix: 'rate-limit:',
        // Optional settings
        windowMs: 15 * 60 * 1000, // 15 minutes
        max: 100 // limit each IP to 100 requests per windowMs
    }),
    message: {
        status: 'error',
        message: 'Too many requests from this IP, please try again after 15 minutes'
    },
    standardHeaders: true, // Return rate limit info in the `RateLimit-*` headers
    legacyHeaders: false // Disable the `X-RateLimit-*` headers
});

// Create specific limiters for different routes
const authLimiter = rateLimit({
    store: new RedisStore({
        client: redisService.client,
        prefix: 'auth-limit:'
    }),
    windowMs: 60 * 60 * 1000, // 1 hour
    max: 5, // limit each IP to 5 login attempts per hour
    message: {
        status: 'error',
        message: 'Too many login attempts from this IP, please try again after an hour'
    }
});

const newsletterLimiter = rateLimit({
    store: new RedisStore({
        client: redisService.client,
        prefix: 'newsletter-limit:'
    }),
    windowMs: 24 * 60 * 60 * 1000, // 24 hours
    max: 3, // limit each IP to 3 newsletter subscriptions per day
    message: {
        status: 'error',
        message: 'Too many subscription attempts from this IP, please try again tomorrow'
    }
});

module.exports = {
    limiter,
    authLimiter,
    newsletterLimiter
}; 
 