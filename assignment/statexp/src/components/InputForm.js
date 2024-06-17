
import React, { useState } from 'react';

const InputForm = () => {
  const [inputs, setInputs] = useState({
    field1: '',
    field2: '',
    field3: '',
    field4: '',
    field5: ''
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setInputs(prevInputs => ({ ...prevInputs, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log(inputs);
  };

  return (
    <form onSubmit={handleSubmit}>
      {Object.keys(inputs).map((field, idx) => (
        <div key={idx}>
          <label>{field}:</label>
          <input
            type="text"
            name={field}
            value={inputs[field]}
            onChange={handleChange}
          />
        </div>
      ))}
      <button type="submit">Submit</button>
    </form>
  );
};

export default InputForm;
