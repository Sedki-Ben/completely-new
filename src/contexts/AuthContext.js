import React, { createContext, useState, useContext, useEffect } from 'react';
import api from '../services/api';
import axios from 'axios';

const AuthContext = createContext(null);

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [isAuthenticated, setIsAuthenticated] = useState(false);

    useEffect(() => {
        checkAuth();
    }, []);

    const checkAuth = async () => {
        try {
            const token = localStorage.getItem('token');
            if (token) {
                axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
            const response = await api.get('/auth/me');
            setUser(response.data);
                setIsAuthenticated(true);
            }
        } catch (err) {
            console.error('Auth check failed:', err);
            localStorage.removeItem('token');
            delete axios.defaults.headers.common['Authorization'];
        } finally {
            setLoading(false);
        }
    };

    const login = async (email, password) => {
        try {
            setError(null);
            const response = await api.post('/auth/login', { email, password });
            const { token, user: userData } = response.data;
            
            localStorage.setItem('token', token);
            axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
            setUser(userData);
            setIsAuthenticated(true);
            
            return userData;
        } catch (err) {
            setError(err.response?.data?.msg || 'Login failed');
            throw err;
        }
    };

    const register = async (userData) => {
        try {
            setError(null);
            const response = await api.post('/auth/register', userData);
            const { token, user: newUser } = response.data;
            
            localStorage.setItem('token', token);
            axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
            setUser(newUser);
            setIsAuthenticated(true);
            
            return newUser;
        } catch (err) {
            setError(err.response?.data?.msg || 'Registration failed');
            throw err;
        }
    };

    const logout = async () => {
        try {
            await api.post('/auth/logout');
        } catch (err) {
            console.error('Logout error:', err);
        } finally {
            localStorage.removeItem('token');
            delete axios.defaults.headers.common['Authorization'];
            setUser(null);
            setIsAuthenticated(false);
        }
    };

    const updateProfile = async (formData) => {
        try {
            console.log('Updating profile with data:', formData);
            const response = await api.put('/auth/profile', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            console.log('Profile update response:', response.data);
            setUser(response.data);
            return response.data;
        } catch (err) {
            console.error('Profile update error:', err);
            setError(err.response?.data?.msg || 'Profile update failed');
            throw err;
        }
    };

    const changePassword = async (currentPassword, newPassword) => {
        try {
            setError(null);
            await api.put('/auth/password', { currentPassword, newPassword });
        } catch (err) {
            setError(err.response?.data?.msg || 'Password change failed');
            throw err;
        }
    };

    const forgotPassword = async (email) => {
        try {
            setError(null);
            await api.post('/auth/forgot-password', { email });
        } catch (err) {
            setError(err.response?.data?.msg || 'Password reset request failed');
            throw err;
        }
    };

    const resetPassword = async (token, password) => {
        try {
            setError(null);
            await api.post(`/auth/reset-password/${token}`, { password });
        } catch (err) {
            setError(err.response?.data?.msg || 'Password reset failed');
            throw err;
        }
    };

    const value = {
        user,
        loading,
        error,
        isAuthenticated,
        login,
        register,
        logout,
        updateProfile,
        changePassword,
        forgotPassword,
        resetPassword,
        isWriter: user?.role === 'writer' || user?.role === 'admin',
        isAdmin: user?.role === 'admin'
    };

    return (
        <AuthContext.Provider value={value}>
            {!loading && children}
        </AuthContext.Provider>
    );
};

export default AuthContext; 