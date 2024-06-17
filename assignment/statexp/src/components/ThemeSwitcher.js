
import React from 'react';

const ThemeSwitcher = ({ isDark, toggleTheme }) => {
  return (
    <button onClick={toggleTheme}>
      Switch to {isDark ? 'Light' : 'Dark'} Mode
    </button>
  );
};

export default ThemeSwitcher;
