import React from 'react';
import { Link } from 'react-router-dom';
import { FiMail, FiTwitter, FiLinkedin } from 'react-icons/fi';

function About() {
  const stats = [
    { label: "Articles Published", value: "150+" },
    { label: "Years Experience", value: "8" },
    { label: "Matches Analyzed", value: "500+" },
    { label: "Reader Countries", value: "45" }
  ];

  return (
    <div className="max-w-4xl mx-auto py-8">
      <h1 className="text-3xl font-bold mb-8">About My Football Journal</h1>

      <section className="bg-white p-8 rounded-lg shadow-md mb-8">
        <h2 className="text-2xl font-semibold mb-4">Our Mission</h2>
        <p className="text-gray-700 mb-6">
          My Football Journal was established in 2016 with a simple mission: to provide in-depth, 
          analytical coverage of football that goes beyond the surface-level analysis typically 
          found in mainstream media. We combine tactical knowledge with data analytics to offer 
          unique insights into the beautiful game.
        </p>
        <p className="text-gray-700 mb-6">
          Our content spans from detailed match analyses to long-form articles about the cultural 
          impact of football around the world. We believe in quality over quantity, ensuring each 
          piece provides valuable insights to our readers.
        </p>
      </section>

      <section className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        {stats.map((stat, index) => (
          <div key={index} className="bg-white p-6 rounded-lg shadow-md text-center">
            <div className="text-3xl font-bold text-blue-900 mb-2">{stat.value}</div>
            <div className="text-gray-600">{stat.label}</div>
          </div>
        ))}
      </section>

      <section className="bg-white p-8 rounded-lg shadow-md mb-8">
        <h2 className="text-2xl font-semibold mb-4">The Team</h2>
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-medium mb-2">John Smith</h3>
            <p className="text-gray-600 mb-4">Founder & Lead Analyst</p>
            <p className="text-gray-700">
              Former youth coach and UEFA B licensed analyst with experience working 
              in professional football. Graduate of the Football Tactics & Analysis 
              program at the International Football Institute.
            </p>
          </div>
        </div>
      </section>

      <section className="bg-white p-8 rounded-lg shadow-md">
        <h2 className="text-2xl font-semibold mb-4">Get in Touch</h2>
        <div className="flex space-x-4">
          <a href="mailto:contact@myfootballjournal.com" className="flex items-center text-gray-700 hover:text-blue-900">
            <FiMail className="mr-2" /> Email
          </a>
          <a href="https://twitter.com/footjournal" className="flex items-center text-gray-700 hover:text-blue-900">
            <FiTwitter className="mr-2" /> Twitter
          </a>
          <a href="https://linkedin.com/company/myfootballjournal" className="flex items-center text-gray-700 hover:text-blue-900">
            <FiLinkedin className="mr-2" /> LinkedIn
          </a>
        </div>
      </section>

      <div className="mt-8 text-center">
        <Link to="/" className="text-blue-900 hover:underline">
          Back to Home
        </Link>
      </div>
    </div>
  );
}

export default About;

