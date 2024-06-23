const cssFilePath = 'styles.css';

fs.stat(cssFilePath, (err, stats) => {
    if (!err && stats.isFile()) {
        fs.unlink(cssFilePath, (err) => {
            if (err) {
                console.error(`Error deleting ${cssFilePath}:`, err);
            } else {
                console.log(`${cssFilePath} deleted successfully!`);
            }
        });
    } else {
        console.log(`${cssFilePath} does not exist or is not a file.`);
    }
});
