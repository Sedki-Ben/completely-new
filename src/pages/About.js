import React from 'react';
import { Link } from 'react-router-dom';
import { FiMail, FiTwitter, FiLinkedin } from 'react-icons/fi';
import { useTranslation } from 'react-i18next';
import Newsletter from '../components/Newsletter';

function About() {
  const { t } = useTranslation();
  
  const stats = [
    { label: "Articles Published", value: "150+" },
    { label: "Years Experience", value: "8" },
    { label: "Matches Analyzed", value: "500+" },
    { label: "Reader Countries", value: "45" }
  ];

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-serif font-bold text-gray-900 dark:text-white mb-8">
        {t('About')}
      </h1>

      <section className="bg-white dark:bg-gray-800 p-8 rounded-xl shadow-md mb-8">
        <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
          Our Mission
        </h2>
        <p className="text-gray-700 dark:text-gray-300 mb-6">
          My Football Journal was established in 2016 with a simple mission: to provide in-depth, 
          analytical coverage of football that goes beyond the surface-level analysis typically 
          found in mainstream media. We combine tactical knowledge with data analytics to offer 
          unique insights into the beautiful game.
        </p>
        <p className="text-gray-700 dark:text-gray-300 mb-6">
          Our content spans from detailed match analyses to long-form articles about the cultural 
          impact of football around the world. We believe in quality over quantity, ensuring each 
          piece provides valuable insights to our readers.
        </p>
      </section>

      <section className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        {stats.map((stat, index) => (
          <div key={index} className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-md text-center">
            <div className="text-3xl font-bold text-slate-700 dark:text-slate-400 mb-2">{stat.value}</div>
            <div className="text-gray-600 dark:text-gray-400">{stat.label}</div>
          </div>
        ))}
      </section>

      <section className="bg-white dark:bg-gray-800 p-8 rounded-xl shadow-md mb-8">
        <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
          The Team
        </h2>
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-medium text-gray-900 dark:text-white mb-2">John Smith</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">Founder & Lead Analyst</p>
            <p className="text-gray-700 dark:text-gray-300">
              Former youth coach and UEFA B licensed analyst with experience working 
              in professional football. Graduate of the Football Tactics & Analysis 
              program at the International Football Institute.
            </p>
          </div>
        </div>
      </section>

      <section className="bg-white dark:bg-gray-800 p-8 rounded-xl shadow-md mb-8">
        <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">
          Get in Touch
        </h2>
        <div className="flex space-x-4">
          <a href="mailto:contact@myfootballjournal.com" className="flex items-center text-gray-700 dark:text-gray-300 hover:text-slate-700 dark:hover:text-slate-400">
            <FiMail className="mr-2" /> Email
          </a>
          <a href="https://twitter.com/footjournal" className="flex items-center text-gray-700 dark:text-gray-300 hover:text-slate-700 dark:hover:text-slate-400">
            <FiTwitter className="mr-2" /> Twitter
          </a>
          <a href="https://linkedin.com/company/myfootballjournal" className="flex items-center text-gray-700 dark:text-gray-300 hover:text-slate-700 dark:hover:text-slate-400">
            <FiLinkedin className="mr-2" /> LinkedIn
          </a>
        </div>
      </section>

      {/* Newsletter Section */}
      <div className="mb-8">
        <Newsletter variant="default" />
      </div>
    </div>
  );
}

export default About; 