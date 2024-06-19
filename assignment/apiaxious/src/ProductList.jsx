
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './ProductList.css';

const ProductList = () => {
  const [products, setProducts] = useState([]);
  const [joke, setJoke] = useState('');

  useEffect(() => {
    const fetchProducts = async () => {
      try {
        const response = await axios.get('https://dummyjson.com/products');
        setProducts(response.data.products);
      } catch (error) {
        console.error('Error fetching products:', error);
      }
    };

    const fetchJoke = async () => {
      try {
        const response = await axios.get('https://official-joke-api.appspot.com/random_joke');
        setJoke(response.data);
      } catch (error) {
        console.error('Error fetching joke:', error);
      }
    };

    fetchProducts();
    fetchJoke();
  }, []);

  return (
    <div className="productlist">
      <h1>Products</h1>
      <ul>
        {products.map(product => (
          <li key={product.id}>
            <h2>{product.title}</h2>
            <p>{product.description}</p>
            <p><strong>Price:</strong> ${product.price}</p>
          </li>
        ))}
      </ul>
      <h1>Random Joke</h1>
      <p><strong>{joke.setup}</strong> - {joke.punchline}</p>
    </div>
  );
};

export default ProductList;
