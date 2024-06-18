import React, { useState } from 'react';

function Todo() {
  const [text, setText] = useState('');
  const [todo, setTodo] = useState([]);

  const addTodo = () => {
    let newTodo = [...todo];

    if (todo.length >= 1) {
      let newId = todo[todo.length - 1].id + 1;
      newTodo.push({ id: newId, task: text, isComplete: false, isEditing: false });
    } else {
      newTodo.push({ id: 1, task: text, isComplete: false, isEditing: false });
    }

    setTodo(newTodo);
    setText('');
  };
  let deleteTast = (i) => {
    let newTodo = [...todo]
    newTodo.splice(i, 1)
    setTodo(newTodo)

    // setTodo(todo.filter((v,index)=>index!=i))
}

  const editTask = (index) => {
    setTodo(todo.map((item, i) => 
      i === index ? { ...item, isEditing: true } : { ...item, isEditing: false }
    ));
    setText(todo[index].task);
  };

  const handleInputChange = (event) => {
    setText(event.target.value);
  };

  const saveTask = (index) => {
    setTodo(todo.map((item, i) => 
      i === index ? { ...item, task: text, isEditing: false } : item
    ));
    setText('');
  };

  const taskComplete = (index) => {
    let newTodo = [...todo];
    newTodo[index].isComplete = true;
    setTodo(newTodo);
  };

  return (
    <>
      <div>
        <input type="text" onChange={(e) => setText(e.target.value)} value={text} />
        <button onClick={addTodo}>Add</button>
      </div>
      <ol>
        {todo.map((item, index) => (
          <li key={item.id}>
            {item.isEditing ? (
              <div>
                <input type="text" onChange={handleInputChange} value={text} />
                <button onClick={() => saveTask(index)}>Save</button>
              </div>
            ) : (
              <div>
                <span style={{ color: item.isComplete ? 'green' : 'red' }}>{item.task}</span>
                <button onClick={() => taskComplete(index)}>Complete</button>
                <button onClick={() => editTask(index)}>Edit</button>
                <button onClick={() => deleteTast(index)}>Delete</button>
              </div>
            )}
          </li>
        ))}
      </ol>
    </>
  );
}

export default Todo;
