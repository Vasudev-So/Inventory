// src/App.js
import React from 'react';
import './App.css';
import { packagingItems } from './packagingItems';
import PackagingList from './PackagingList';

const App = () => {
  const isDark = true; // Change to false to see the light theme
  const isOpen = true; // Change to false to hide the PackagingList component

  const appStyle = {
    backgroundColor: isDark ? '#333' : '#fff',
    color: isDark ? '#fff' : '#000',
    minHeight: '100vh',
    padding: '20px',
  };

  return (
    <div className="App" style={appStyle}>
      <h1>Packaging Items</h1>
      {isOpen && <PackagingList items={packagingItems} />}
    </div>
  );
};

export default App;
