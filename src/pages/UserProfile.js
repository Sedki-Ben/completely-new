import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useTranslation } from 'react-i18next';
import defaultMaleAvatar from '../assets/images/mann.png';
import defaultFemaleAvatar from '../assets/images/frau.png';
import { FiEdit2, FiSave, FiX, FiCamera, FiTrash2 } from 'react-icons/fi';
import { getUserAvatarUrl } from '../utils/imageUtils';

const UserProfile = () => {
  const { t } = useTranslation();
  const { user, updateProfile } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [profileImage, setProfileImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [imageError, setImageError] = useState('');
  
  const [formData, setFormData] = useState({
    name: '',
    dateOfBirth: '',
    gender: '',
    location: '',
    bio: '',
    profession: '',
    website: '',
    twitter: '',
    linkedin: ''
  });

  // Maximum file size (5MB)
  const MAX_FILE_SIZE = 5 * 1024 * 1024;
  const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];

  useEffect(() => {
    if (user) {
      // Format date properly for date input
      const formatDateForInput = (dateString) => {
        if (!dateString) return '';
        const date = new Date(dateString);
        if (isNaN(date.getTime())) return '';
        return date.toISOString().split('T')[0];
      };

      setFormData(prev => ({
        ...prev,
        name: user.name || '',
        dateOfBirth: formatDateForInput(user.dateOfBirth),
        gender: user.gender || '',
        location: user.location || '',
        bio: user.bio || '',
        profession: user.profession || '',
        website: user.website || '',
        twitter: user.twitter || '',
        linkedin: user.linkedin || ''
      }));
      
      // Use utility function for avatar URL
      setPreviewUrl(getUserAvatarUrl(user));
    }
  }, [user]);

  // Clear success message after 3 seconds
  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(false), 3000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  const validateImage = (file) => {
    if (!file) return true;
    
    if (!ALLOWED_TYPES.includes(file.type)) {
      return t('Please select a valid image file (JPEG, PNG, or WebP)');
    }
    
    if (file.size > MAX_FILE_SIZE) {
      return t('Image size must be less than 5MB');
    }
    
    return true;
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImageError('');
    
    if (file) {
      const validation = validateImage(file);
      if (validation !== true) {
        setImageError(validation);
        e.target.value = ''; // Clear the input
        return;
      }
      
      setProfileImage(file);
      
      // Create preview URL and handle cleanup
      const objectUrl = URL.createObjectURL(file);
      setPreviewUrl(objectUrl);
      
      // Cleanup previous object URL
      return () => URL.revokeObjectURL(objectUrl);
    }
  };

  const handleRemoveImage = () => {
    setProfileImage(null);
    
    // Reset to default avatar
    const defaultAvatar = user?.gender === 'female' ? defaultFemaleAvatar : defaultMaleAvatar;
    setPreviewUrl(defaultAvatar);
    setImageError('');
    
    // Clear file input
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) fileInput.value = '';
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    const newFormData = {
      ...formData,
      [name]: value
    };
    setFormData(newFormData);
    
    // Update preview URL if gender changes and no custom image is selected
    if (name === 'gender' && !profileImage) {
      const updatedUser = { ...user, gender: value };
      setPreviewUrl(getUserAvatarUrl(updatedUser));
    }
    
    // Clear error when user starts typing
    if (error) setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess(false);

    try {
      const formDataToSend = new FormData();
      
      if (profileImage) {
        const validation = validateImage(profileImage);
        if (validation !== true) {
          setError(validation);
          setLoading(false);
          return;
        }
        formDataToSend.append('profileImage', profileImage);
      }
      
      Object.keys(formData).forEach(key => {
        if (formData[key]) {
          formDataToSend.append(key, formData[key]);
        }
      });

      await updateProfile(formDataToSend);
      setSuccess(true);
      setIsEditing(false);
      setProfileImage(null); // Clear the file state after successful upload
    } catch (err) {
      setError(err.response?.data?.msg || t('Failed to update profile'));
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    setIsEditing(false);
    setError('');
    setImageError('');
    setProfileImage(null);
    
    // Reset form data to original user data
    if (user) {
      // Format date properly for date input
      const formatDateForInput = (dateString) => {
        if (!dateString) return '';
        const date = new Date(dateString);
        if (isNaN(date.getTime())) return '';
        return date.toISOString().split('T')[0];
      };

      setFormData({
        name: user.name || '',
        dateOfBirth: formatDateForInput(user.dateOfBirth),
        gender: user.gender || '',
        location: user.location || '',
        bio: user.bio || '',
        profession: user.profession || '',
        website: user.website || '',
        twitter: user.twitter || '',
        linkedin: user.linkedin || ''
      });
      
      // Use utility function for avatar URL
      setPreviewUrl(getUserAvatarUrl(user));
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-100 via-blue-100 to-purple-100 dark:from-gray-900 dark:via-slate-900 dark:to-gray-800 transition-colors duration-500 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white/80 dark:bg-slate-900/80 backdrop-blur-lg rounded-2xl shadow-xl p-8">
          <div className="flex justify-between items-center mb-8">
            <h1 className="text-3xl font-bold text-gray-800 dark:text-gray-100">{t('Profile')}</h1>
            <button
              onClick={() => isEditing ? handleCancel() : setIsEditing(true)}
              className="flex items-center px-4 py-2 rounded-lg bg-blue-500 text-white hover:bg-blue-600 transition disabled:opacity-50"
              disabled={loading}
            >
              {isEditing ? (
                <>
                  <FiX className="mr-2" />
                  {t('Cancel')}
                </>
              ) : (
                <>
                  <FiEdit2 className="mr-2" />
                  {t('Edit Profile')}
                </>
              )}
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Profile Image Section */}
            <div className="flex flex-col items-center space-y-4">
              <div className="relative group">
                <img
                  src={previewUrl}
                  alt={formData.name || t('Profile')}
                  className="w-32 h-32 rounded-full object-cover border-4 border-white dark:border-gray-800 shadow-lg"
                  onError={(e) => {
                    e.target.src = formData.gender === 'female' ? defaultFemaleAvatar : defaultMaleAvatar;
                  }}
                />
                {isEditing && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 rounded-full opacity-0 group-hover:opacity-100 transition-opacity">
                    <div className="flex gap-2">
                      <label className="bg-blue-500 text-white p-2 rounded-full cursor-pointer hover:bg-blue-600 transition">
                        <FiCamera className="w-4 h-4" />
                        <input
                          type="file"
                          accept="image/*"
                          onChange={handleImageChange}
                          className="hidden"
                        />
                      </label>
                      {profileImage && (
                        <button
                          type="button"
                          onClick={handleRemoveImage}
                          className="bg-red-500 text-white p-2 rounded-full hover:bg-red-600 transition"
                        >
                          <FiTrash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>
              
              {imageError && (
                <p className="text-red-500 text-sm text-center">{imageError}</p>
              )}
              
              <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-100">
                {formData.name || t('Your Name')}
              </h2>
              {formData.profession && (
                <p className="text-gray-600 dark:text-gray-400">{formData.profession}</p>
              )}
            </div>

            {/* Form Fields */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {t('Name')} <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  disabled={!isEditing}
                  required
                  className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-700 bg-white/50 dark:bg-slate-800 focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition text-gray-900 dark:text-white"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {t('Date of Birth')}
                </label>
                <input
                  type="date"
                  name="dateOfBirth"
                  value={formData.dateOfBirth}
                  onChange={handleInputChange}
                  disabled={!isEditing}
                  className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-700 bg-white/50 dark:bg-slate-800 focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition text-gray-900 dark:text-white"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {t('Gender')}
                </label>
                <select
                  name="gender"
                  value={formData.gender}
                  onChange={handleInputChange}
                  disabled={!isEditing}
                  className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-700 bg-white/50 dark:bg-slate-800 focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition text-gray-900 dark:text-white"
                >
                  <option value="">{t('Select Gender')}</option>
                  <option value="male">{t('Male')}</option>
                  <option value="female">{t('Female')}</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {t('Location')}
                </label>
                <input
                  type="text"
                  name="location"
                  value={formData.location}
                  onChange={handleInputChange}
                  disabled={!isEditing}
                  className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-700 bg-white/50 dark:bg-slate-800 focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition text-gray-900 dark:text-white"
                />
              </div>

              <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {t('Bio')}
                </label>
                <textarea
                  name="bio"
                  value={formData.bio}
                  onChange={handleInputChange}
                  disabled={!isEditing}
                  rows="4"
                  maxLength="500"
                  className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-700 bg-white/50 dark:bg-slate-800 focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition text-gray-900 dark:text-white"
                />
                <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                  {formData.bio.length}/500 {t('characters')}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {t('Profession')}
                </label>
                <input
                  type="text"
                  name="profession"
                  value={formData.profession}
                  onChange={handleInputChange}
                  disabled={!isEditing}
                  className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-700 bg-white/50 dark:bg-slate-800 focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition text-gray-900 dark:text-white"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {t('Website')}
                </label>
                <input
                  type="url"
                  name="website"
                  value={formData.website}
                  onChange={handleInputChange}
                  disabled={!isEditing}
                  placeholder="https://example.com"
                  className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-700 bg-white/50 dark:bg-slate-800 focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition text-gray-900 dark:text-white"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {t('Twitter')}
                </label>
                <input
                  type="text"
                  name="twitter"
                  value={formData.twitter}
                  onChange={handleInputChange}
                  disabled={!isEditing}
                  placeholder="@username"
                  className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-700 bg-white/50 dark:bg-slate-800 focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition text-gray-900 dark:text-white"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {t('LinkedIn')}
                </label>
                <input
                  type="text"
                  name="linkedin"
                  value={formData.linkedin}
                  onChange={handleInputChange}
                  disabled={!isEditing}
                  placeholder="linkedin.com/in/username"
                  className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-700 bg-white/50 dark:bg-slate-800 focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition text-gray-900 dark:text-white"
                />
              </div>
            </div>

            {/* Error and Success Messages */}
            {error && (
              <div className="text-red-600 text-center font-medium p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                {error}
              </div>
            )}
            {success && (
              <div className="text-green-600 text-center font-medium p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                {t('Profile updated successfully')}
              </div>
            )}

            {/* Submit Button */}
            {isEditing && (
              <div className="flex justify-end gap-3">
                <button
                  type="button"
                  onClick={handleCancel}
                  disabled={loading}
                  className="px-6 py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 focus:ring-4 focus:ring-gray-500/50 disabled:opacity-50 transition"
                >
                  {t('Cancel')}
                </button>
                <button
                  type="submit"
                  disabled={loading || !formData.name.trim()}
                  className="flex items-center px-6 py-3 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 focus:ring-4 focus:ring-emerald-500/50 disabled:opacity-50 transition"
                >
                  <FiSave className="mr-2" />
                  {loading ? t('Saving...') : t('Save Changes')}
                </button>
              </div>
            )}
          </form>
        </div>
      </div>
    </div>
  );
};

export default UserProfile;