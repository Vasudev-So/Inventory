import React, { useState } from 'react';

const PackagingList = () => {
  const [items, setItems] = useState([
    { id: 1, item: 'Shoes', isPacked: true, isHeavy: false },
  ]);
  const [newItem, setNewItem] = useState('');
  const [isPacked, setIsPacked] = useState(false);
  const [isHeavy, setIsHeavy] = useState(false);

  const addItem = () => {
    const id = items.length ? items[items.length - 1].id + 1 : 1;
    setItems([
      ...items,
      { id, item: newItem, isPacked, isHeavy },
    ]);
    setNewItem('');
    setIsPacked(false);
    setIsHeavy(false);
  };

  return (
    <div>
      <h1>Packaging List</h1>
      <input
        type="text"
        value={newItem}
        onChange={(e) => setNewItem(e.target.value)}
        placeholder="Item name"
      />
      <label>
        <input
          type="checkbox"
          checked={isPacked}
          onChange={(e) => setIsPacked(e.target.checked)}
        />
        Packed
      </label>
      <label>
        <input
          type="checkbox"
          checked={isHeavy}
          onChange={(e) => setIsHeavy(e.target.checked)}
        />
        Heavy
      </label>
      <button onClick={addItem}>Add Item</button>
      <ul>
        {items.map(({ id, item, isPacked, isHeavy }) => (
          <li
            key={id}
            style={{
              backgroundColor: isPacked ? 'green' : isHeavy ? 'black' : 'white',
              color: isHeavy ? 'white' : 'black',
            }}
          >
            {item} - {isPacked ? 'Packed' : 'Not Packed'}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default PackagingList;
