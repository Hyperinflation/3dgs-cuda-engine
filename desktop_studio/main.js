const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { exec, spawn } = require('child_process');
const fs = require('fs');

// GPU Hardware Acceleration for RTX 3090
app.commandLine.appendSwitch('enable-gpu-rasterization');
app.commandLine.appendSwitch('enable-zero-copy');
app.commandLine.appendSwitch('ignore-gpu-blocklist');
app.commandLine.appendSwitch('high-dpi-support', '1');

let mainWindow;
let trainingProcess = null;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1460,
        height: 920,
        minWidth: 1100,
        minHeight: 720,
        title: "Postshot Studio Pro - 3D Gaussian Splatting Mimari Engine",
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
        if (trainingProcess) {
            try { trainingProcess.kill(); } catch (e) {}
        }
    });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

// IPC: 1-Click ADB Phone Mirroring
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

// IPC: Postshot CUDA Training
ipcMain.on('start-training', (event, args) => {
    if (trainingProcess) return;

    const rootDir = path.resolve(__dirname, '..');
    const pythonPath = path.join(rootDir, 'venv', 'Scripts', 'python.exe');
    const scriptPath = path.join(__dirname, 'engine', 'trainer.py');
    const iters = args.iterations || 30000;

    if (!fs.existsSync(pythonPath)) {
        if (mainWindow) {
            mainWindow.webContents.send('training-log', `\n[HATA] Python ortamı bulunamadı: ${pythonPath}`);
            mainWindow.webContents.send('training-finished', 1);
        }
        return;
    }

    try {
        trainingProcess = spawn(pythonPath, [scriptPath, '--iterations', iters.toString()], {
            cwd: rootDir,
            windowsHide: true
        });

        trainingProcess.stdout.on('data', (data) => {
            if (mainWindow) {
                mainWindow.webContents.send('training-log', data.toString());
            }
        });

        trainingProcess.stderr.on('data', (data) => {
            if (mainWindow) {
                mainWindow.webContents.send('training-log', data.toString());
            }
        });

        trainingProcess.on('error', (err) => {
            if (mainWindow) {
                mainWindow.webContents.send('training-log', `\n[HATA] İşlem başlatılamadı: ${err.message}`);
                mainWindow.webContents.send('training-finished', 1);
            }
            trainingProcess = null;
        });

        trainingProcess.on('close', (code) => {
            trainingProcess = null;
            if (mainWindow) {
                mainWindow.webContents.send('training-finished', code || 0);
            }
        });
    } catch (err) {
        if (mainWindow) {
            mainWindow.webContents.send('training-log', `\n[HATA] ${err.message}`);
            mainWindow.webContents.send('training-finished', 1);
        }
        trainingProcess = null;
    }
});

ipcMain.on('stop-training', () => {
    if (trainingProcess) {
        try { trainingProcess.kill(); } catch (e) {}
        trainingProcess = null;
        if (mainWindow) {
            mainWindow.webContents.send('training-log', "\n[!] Eğitim kullanıcı tarafından durduruldu.\n");
        }
    }
});
