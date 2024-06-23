const fs = require('fs');
fs.readFile('index.html', 'utf8', (err, data) => {
    if (err) {
        console.error('Error reading file:', err);
        return;
    }
    let mData = data.replace(/<title>.*<\/title>/, '<title>fs module</title>');
    fs.writeFile('index.html', mData, 'utf8', (err) => {
        if (err) {
            console.error('Error writing file:', err);
            return;
        }
        console.log('index.html updated successfully!');
    });
});
