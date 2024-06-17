
import React, { useState } from 'react';
import InputForm from './components/InputForm';
import Counter from './components/Counter';
import ThemeSwitcher from './components/ThemeSwitcher';
import ToggleComponent from './components/ToggleComponent';
import './App.css';

co
  const [isDark, setIsDark] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  const toggleTheme = () => setIsDark(prevIsDark => !prevIsDark);
  const toggleComponent = () => setIsOpen(prevIsOpen => !prevIsOpen);

  return (
    <div className={`app ${isDark ? 'dark' : 'light'}`}>
      <h1>React App</h1>
      <InputForm />
      <Counter />
      <ThemeSwitcher isDark={isDark} toggleTheme={toggleTheme} />
      <button onClick={toggleComponent}>
        {isOpen ? 'Hide' : 'Show'} Component
      </button>
      {isOpen && <ToggleComponent />}
    </div>
  );
};

export default App;
