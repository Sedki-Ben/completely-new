import React, { useState, useRef, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useTranslation } from 'react-i18next';
import defaultMaleAvatar from '../assets/images/mann.png';
import defaultFemaleAvatar from '../assets/images/frau.png';
import { FiUser, FiLogOut, FiSettings, FiChevronDown, FiLayout } from 'react-icons/fi';
import { getUserAvatarUrl } from '../utils/imageUtils';

const UserAvatar = () => {
  const { user, isAuthenticated, logout } = useAuth();
  const { t } = useTranslation();
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [avatarUrl, setAvatarUrl] = useState('');
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Update avatar URL when user data changes
  useEffect(() => {
    if (user) {
      const newAvatarUrl = getUserAvatarUrl(user);
      console.log('User data changed, updating avatar URL:', { 
        userProfileImage: user.profileImage, 
        newAvatarUrl, 
        userGender: user.gender 
      });
      setAvatarUrl(newAvatarUrl);
    }
  }, [user]);

  const handleLogout = async () => {
    try {
      await logout();
      setIsDropdownOpen(false);
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  if (!isAuthenticated) {
    return (
      <Link 
        to="/signin" 
        className="flex items-center px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-200 hover:text-emerald-600 dark:hover:text-emerald-400 transition-colors duration-300"
      >
        <FiUser className="w-4 h-4 mr-2" />
        {t('Sign In')}
      </Link>
    );
  }

  return (
    <div className="relative" ref={dropdownRef} dir="ltr">
      <button
        onClick={() => setIsDropdownOpen(!isDropdownOpen)}
        className="flex items-center gap-2 group focus:outline-none"
      >
        <div className="relative">
          <img
            src={avatarUrl}
            alt={user?.name || 'Profile'}
            className="w-10 h-10 rounded-full object-cover border-2 border-transparent group-hover:border-emerald-500 transition-all duration-300 shadow-md"
            onError={(e) => {
              e.target.src = user?.gender === 'female' ? defaultFemaleAvatar : defaultMaleAvatar;
            }}
          />
          <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-emerald-500 rounded-full border-2 border-white dark:border-gray-900"></div>
        </div>
        <span className="font-medium text-gray-700 dark:text-gray-200 group-hover:text-emerald-500 transition-colors duration-300">
          {user?.name}
        </span>
        <FiChevronDown className={`w-4 h-4 text-gray-500 transition-transform duration-300 ${isDropdownOpen ? 'rotate-180' : ''}`} />
      </button>

      {/* Dropdown Menu */}
      {isDropdownOpen && (
        <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-lg shadow-lg py-1 z-50 border border-gray-200 dark:border-gray-700">
          <Link
            to="/profile"
            className="flex items-center px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-300"
            onClick={() => setIsDropdownOpen(false)}
          >
            <FiSettings className="w-4 h-4 mr-2" />
            {t('Profile Settings')}
          </Link>
          {user?.role === 'admin' && (
            <Link
              to="/admin"
              className="flex items-center px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-300"
              onClick={() => setIsDropdownOpen(false)}
            >
              <FiLayout className="w-4 h-4 mr-2" />
              {t('Dashboard')}
            </Link>
          )}
          <button
            onClick={handleLogout}
            className="w-full flex items-center px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-300"
          >
            <FiLogOut className="w-4 h-4 mr-2" />
            {t('Sign Out')}
          </button>
        </div>
      )}
    </div>
  );
};

export default UserAvatar; 