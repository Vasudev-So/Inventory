const userDetails = {
    name: 'John Doe',
    email: 'john.doe@example.com',
    phone: '123-456-7890'
};

function replacePlaceholders(filePath, userDetails) {
    fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) {
            console.error(`Error reading file ${filePath}:`, err);
            return;
        }

        let updatedData = data;
        Object.keys(userDetails).forEach(key => {
            updatedData = updatedData.replace(`{%%${key}%%}`, userDetails[key]);
        });

        fs.writeFile(filePath, updatedData, 'utf8', (err) => {
            if (err) {
                console.error(`Error writing file ${filePath}:`, err);
            } else {
                console.log(`${filePath} updated with user details successfully!`);
            }
        });
    });
}

replacePlaceholders('user_details.html', userDetails);
