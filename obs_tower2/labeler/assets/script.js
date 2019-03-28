const checkboxes = Array.prototype.slice.apply(document.getElementsByTagName('input'));
const screenshot = document.getElementById('screenshot');
const nameLabel = document.getElementById('name');

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

async function loadNewSample() {
    const name = await (await fetch('/sample')).text();
    const classes = await (await fetch('/classes/' + name)).json();
    checkboxes.forEach((b) => b.checked = false);
    classes.forEach((c, i) => checkboxes[i].checked = c);
    screenshot.src = '/frame/' + name;
    nameLabel.textContent = name;
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

async function saveLabels() {
    const labelData = checkboxes.map((b) => b.checked ? '1' : '0').join(',');
    const path = '/save/' + nameLabel.textContent + '/' + labelData;
    await fetch(path);
    await loadNewSample();
}

window.addEventListener('load', loadNewSample);
window.addEventListener('keypress', keyPressed);
document.getElementById('save-button').addEventListener('click', saveLabels);
