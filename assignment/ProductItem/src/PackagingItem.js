// src/PackagingItem.js
import React from 'react';
import './PackagingItem.css';

const PackagingItem = ({ item }) => {
  const itemStyle = {
    backgroundColor: item.isPacked ? 'green' : item.isHeavy ? 'black' : 'transparent',
    color: item.isHeavy ? 'white' : 'black',
  };

  return (
    <div className="packaging-item" style={itemStyle}>
      <p>{item.item}</p>
      <p>{item.isPacked ? 'Packed ✔️' : 'Not Packed'}</p>
    </div>
  );
};

export default PackagingItem;
