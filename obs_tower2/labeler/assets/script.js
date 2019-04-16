const checkboxes = Array.prototype.slice.apply(document.getElementsByTagName('input'));
const fractions = Array.prototype.slice.apply(document.getElementsByClassName('fraction'));
const screenshot = document.getElementById('screenshot');
const nameLabel = document.getElementById('name');
const getsKey = document.getElementById('gets-key');
let previousName = null;

const KEY_C = 99
const KEY_L = 108;
const KEY_B = 98;
const KEY_O = 111;
const KEY_K = 107;
const KEY_X = 120;
const KEY_H = 104;
const KEY_R = 114;
const KEY_G = 103;
const KEY_T = 116;
const KEY_U = 117;
const KEY_ENTER = 13;
const KEY_BACKSPACE = 8;

async function updateForName(name) {
    checkboxes.forEach((box) => box.checked = false);
    fractions.forEach((f) => f.textContent = '');
    screenshot.src = '/frame/' + name;
    const key = await (await fetch('/key/' + name)).json();
    if (key) {
        getsKey.textContent = 'Gets key';
    } else {
        getsKey.textContent = 'Does not get key';
    }
    const classifications = await (await fetch('/classify/' + name)).json();
    if (classifications) {
        for (let i = 0; i < classifications.length; ++i) {
            const value = classifications[i];
            fractions[i].textContent = '(' + value.toFixed(3) + ')';
            if (value > 0.5) {
                fractions[i].className = 'fraction present';
            } else {
                fractions[i].className = 'fraction absent';
            }
        }
    }
}

async function loadNewSample() {
    const name = await (await fetch('/sample')).text();
    previousName = nameLabel.textContent || null;
    nameLabel.textContent = name;
    await updateForName(name);
}

async function goToPrevious() {
    if (!previousName) {
        return;
    }
    nameLabel.textContent = previousName;
    previousName = null;
    await updateForName(nameLabel.textContent);
}

function keyPressed(event) {
    if (event.which == KEY_ENTER) {
        saveLabels();
        return;
    }
    const boxKeys = [KEY_C, KEY_L, KEY_B, KEY_O, KEY_K, KEY_X, KEY_H, KEY_R, KEY_G, KEY_T, KEY_U];
    const idx = boxKeys.indexOf(event.which);
    if (idx >= 0) {
        checkboxes[idx].checked = !checkboxes[idx].checked;
    }
}

function keyDown(event) {
    if (event.which == KEY_BACKSPACE) {
        event.preventDefault();
        goToPrevious();
    }
}

async function saveLabels() {
    const labelData = checkboxes.map((b) => b.checked ? '1' : '0').join(',');
    const path = '/save/' + nameLabel.textContent + '/' + labelData;
    await fetch(path);
    await loadNewSample();
}

window.addEventListener('load', loadNewSample);
window.addEventListener('keypress', keyPressed);
window.addEventListener('keydown', keyDown);
document.getElementById('save-button').addEventListener('click', saveLabels);
