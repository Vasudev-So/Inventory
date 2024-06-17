import React, { useState } from 'react';

const Counter = () => {
  const [count, setCount] = useState(0);

  const handleChange = (e) => {
    setCount(Number(e.target.value));
  };

  return (
    <div>
      <label>Counter: </label>
      <input type="number" onChange={handleChange} />
      <p>Count: {count}</p>
    </div>
  );
};

export default Counter;
