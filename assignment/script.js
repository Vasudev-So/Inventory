document.getElementById('startButton').addEventListener('click', startBouncing);
document.getElementById('stopButton').addEventListener('click', stopBouncing);

let ball = document.getElementById('ball');
let container = document.getElementById('container');
let bounceInterval;
let gravity = 0.98;
let bounceHeight = 150;
let bounceReduction = 0.9;

function startBouncing() {
    if (bounceInterval) return;

    let height = bounceHeight;
    let position = 0;
    let velocity = 0;
    let isBouncing = true;

    bounceInterval = setInterval(() => {
        if (isBouncing) {
            velocity += gravity;
            position += velocity;

            if (position > container.clientHeight - ball.clientHeight) {
                position = container.clientHeight - ball.clientHeight;
                velocity = -velocity * bounceReduction;
                height *= bounceReduction;

                if (height < 1) {
                    isBouncing = false;
                    clearInterval(bounceInterval);
                    bounceInterval = null;
                }
            }
        } else {
            clearInterval(bounceInterval);
            bounceInterval = null;
        }

        ball.style.top = `${position}px`;
    }, 20);
}

function stopBouncing() {
    clearInterval(bounceInterval);
    bounceInterval = null;
    ball.style.bottom = '0px';
}

