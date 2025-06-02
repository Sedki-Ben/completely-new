import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate, Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { FiEye, FiEyeOff } from 'react-icons/fi';

const SignIn = () => {
  const { t } = useTranslation();
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await login(email, password);
      navigate('/');
    } catch (err) {
      setError(err.response?.data?.msg || t('Login failed'));
    } finally {
      setLoading(false);
    }
  };

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  return (
    <div className="min-h-screen flex items-start justify-center bg-gradient-to-br from-emerald-100 via-blue-100 to-purple-100 dark:from-gray-900 dark:via-slate-900 dark:to-gray-800 transition-colors duration-500 pt-24">
      <div className="relative w-full max-w-md p-8 rounded-2xl shadow-2xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-lg border border-blue-200 dark:border-blue-900">
        {/* Accent bar */}
        <div className="absolute -top-2 left-1/2 -translate-x-1/2 w-32 h-2 rounded bg-gradient-to-r from-emerald-400 via-blue-500 to-purple-500 shadow-md" />
        {/* Logo/Title */}
        <div className="flex flex-col items-center mb-6 mt-2">
          <span className="text-4xl font-extrabold font-serif text-blue-600 dark:text-blue-400 drop-shadow-sm">PTC</span>
          <span className="text-lg font-semibold text-gray-700 dark:text-gray-200 tracking-wide mt-1">{t('Sign in to your account')}</span>
        </div>
        <form onSubmit={handleSubmit} className="space-y-5">
          <input
            type="email"
            className="border border-blue-300 dark:border-blue-700 bg-white/70 dark:bg-slate-800/70 p-3 w-full rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 dark:focus:ring-blue-600 transition placeholder-gray-400 dark:placeholder-gray-500 text-blue-900 dark:text-blue-100"
            placeholder={t('Email')}
            value={email}
            onChange={e => setEmail(e.target.value)}
            required
            dir="auto"
          />
          <div className="relative">
          <input
              type={showPassword ? "text" : "password"}
              className="border border-purple-300 dark:border-purple-700 bg-white/70 dark:bg-slate-800/70 p-3 w-full rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-400 dark:focus:ring-purple-600 transition placeholder-gray-400 dark:placeholder-gray-500 text-purple-900 dark:text-purple-100"
              placeholder={t('Password')}
            value={password}
            onChange={e => setPassword(e.target.value)}
            required
              dir="auto"
          />
            <button
              type="button"
              onClick={togglePasswordVisibility}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 focus:outline-none"
            >
              {showPassword ? <FiEyeOff className="w-5 h-5" /> : <FiEye className="w-5 h-5" />}
            </button>
          </div>
          {error && <div className="text-red-500 text-center font-medium">{error}</div>}
          <button
            type="submit"
            className="bg-gradient-to-r from-blue-500 via-emerald-500 to-purple-500 text-white px-6 py-3 rounded-lg font-semibold shadow-md hover:from-blue-600 hover:to-purple-600 transition w-full disabled:opacity-60"
            disabled={loading}
          >
            {loading ? t('Signing in...') : t('Sign In')}
          </button>
        </form>
        <div className="mt-6 text-center text-gray-600 dark:text-gray-300">
          {t("Don't have an account?")}{' '}
          <Link to="/signup" className="text-emerald-600 dark:text-emerald-400 hover:underline font-semibold">{t('Sign Up')}</Link>
        </div>
      </div>
    </div>
  );
};

export default SignIn; 