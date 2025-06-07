import defaultMaleAvatar from '../assets/images/mann.png';
import defaultFemaleAvatar from '../assets/images/frau.png';

// Get the backend URL
const getBackendUrl = () => {
  return process.env.REACT_APP_API_URL || 'http://localhost:5000';
};

// Convert relative image paths to full URLs
export const getImageUrl = (imagePath) => {
  if (!imagePath) return null;
  
  // If it's already a full URL, return as is
  if (imagePath.startsWith('http')) {
    return imagePath;
  }
  
  // If it's a relative path, construct the full URL
  const backendUrl = getBackendUrl().replace('/api', ''); // Remove /api suffix if present
  return `${backendUrl}${imagePath}`;
};

// Get user avatar URL with fallback to default
export const getUserAvatarUrl = (user) => {
  if (!user) return defaultMaleAvatar;
  
  if (user.profileImage) {
    return getImageUrl(user.profileImage);
  }
  
  // Return default avatar based on gender
  return user.gender === 'female' ? defaultFemaleAvatar : defaultMaleAvatar;
}; 