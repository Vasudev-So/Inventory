
import React from 'react';
import PackagingItem from './PackagingItem';

const PackagingList = ({ items }) => {
  return (
    <div className="packaging-list">
      {items.map(item => (
        <PackagingItem key={item.id} item={item} />
      ))}
    </div>
  );
};

export default PackagingList;
