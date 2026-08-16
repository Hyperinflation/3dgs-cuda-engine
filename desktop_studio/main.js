const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { exec } = require('child_process');
const { runNativeTraining } = require('./engine/native_engine');

// GPU Hardware Acceleration for NVIDIA RTX 3090
app.commandLine.appendSwitch('enable-gpu-rasterization');
app.commandLine.appendSwitch('enable-zero-copy');
app.commandLine.appendSwitch('ignore-gpu-blocklist');
app.commandLine.appendSwitch('high-dpi-support', '1');

let mainWindow;
let isTrainingActive = false;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1460,
        height: 920,
        minWidth: 1100,
        minHeight: 720,
        title: "Postshot Studio Pro - Native C++ & WebGL2 Engine (Zero Python)",
        backgroundColor: "#f5f1ea",
        icon: path.join(__dirname, 'icon.png'),
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
            webSecurity: false
        },
        autoHideMenuBar: true
    });

    mainWindow.loadFile(path.join(__dirname, 'index.html'));

    mainWindow.on('closed', () => {
        mainWindow = null;
        isTrainingActive = false;
    });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

// IPC: 1-Click ADB Phone Mirroring (Pure Native ADB)
ipcMain.handle('mirror-adb', async () => {
    const adbPath = "C:\\Users\\halit\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe";
    return new Promise((resolve) => {
        exec(`"${adbPath}" reverse tcp:8080 tcp:8080`, (err) => {
            exec(`"${adbPath}" shell am start -a android.intent.action.VIEW -d "http://localhost:8080/"`, (err2) => {
                if (err2) {
                    resolve({ success: false, message: err2.message });
                } else {
                    resolve({ success: true, message: "Model Samsung Galaxy cihazınıza başarıyla aktarıldı!" });
                }
            });
        });
    });
});

// IPC: Pure Native 3DGS Training (Zero Python)
ipcMain.on('start-training', (event, args) => {
    if (isTrainingActive) return;
    isTrainingActive = true;

    const iters = args.iterations || 30000;

    try {
        runNativeTraining(iters, 1000, (logText) => {
            if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.send('training-log', logText);
                if (logText.includes('[DONE:')) {
                    isTrainingActive = false;
                    mainWindow.webContents.send('training-finished', 0);
                }
            }
        });
    } catch (err) {
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('training-log', `\n[HATA] ${err.message}`);
            mainWindow.webContents.send('training-finished', 1);
        }
        isTrainingActive = false;
    }
});

ipcMain.on('stop-training', () => {
    isTrainingActive = false;
    if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('training-log', "\n[!] Eğitim durduruldu.\n");
    }
});
