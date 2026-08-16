const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { exec, spawn } = require('child_process');
const fs = require('fs');

// GPU Hardware Acceleration for NVIDIA RTX 3090
app.commandLine.appendSwitch('enable-gpu-rasterization');
app.commandLine.appendSwitch('enable-zero-copy');
app.commandLine.appendSwitch('ignore-gpu-blocklist');
app.commandLine.appendSwitch('high-dpi-support', '1');

let mainWindow;
let trainingProcess = null;
const logFilePath = path.resolve(__dirname, '..', 'training.log');

function appendLog(msg) {
    try {
        fs.appendFileSync(logFilePath, msg + '\n', 'utf8');
    } catch (e) {}
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1460,
        height: 920,
        minWidth: 1100,
        minHeight: 720,
        title: "Postshot Studio Pro - NVIDIA RTX 3090 3DGS Engine",
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

// IPC: Open Log File
ipcMain.handle('open-log-file', async () => {
    if (!fs.existsSync(logFilePath)) {
        fs.writeFileSync(logFilePath, "=== POSTSHOT STUDIO PRO - LOG DOSYASI ===\n", 'utf8');
    }
    exec(`notepad.exe "${logFilePath}"`);
    return { success: true };
});

// IPC: Real RTX 3090 Photometric Loss CUDA Training
ipcMain.on('start-training', (event, args) => {
    if (trainingProcess) return;

    const rootDir = path.resolve(__dirname, '..');
    const pythonPath = path.join(rootDir, 'venv', 'Scripts', 'python.exe');
    const scriptPath = path.join(__dirname, 'engine', 'real_cuda_engine.py');
    const iters = args.iterations || 30000;

    try {
        fs.writeFileSync(logFilePath, `=== POSTSHOT STUDIO PRO - EGITIM BASLATILDI (${new Date().toLocaleString()}) ===\n`, 'utf8');
    } catch (e) {}

    try {
        trainingProcess = spawn(pythonPath, [scriptPath, '--iterations', iters.toString()], {
            cwd: rootDir,
            windowsHide: true
        });

        trainingProcess.stdout.on('data', (data) => {
            const str = data.toString();
            appendLog(str);
            if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.send('training-log', str);
            }
        });

        trainingProcess.stderr.on('data', (data) => {
            const str = data.toString();
            appendLog(str);
            if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.send('training-log', str);
            }
        });

        trainingProcess.on('error', (err) => {
            const str = `\n[HATA] İşlem başlatılamadı: ${err.message}\n`;
            appendLog(str);
            if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.send('training-log', str);
                mainWindow.webContents.send('training-finished', 1);
            }
            trainingProcess = null;
        });

        trainingProcess.on('close', (code) => {
            appendLog(`\n=== EGITIM ISLEMI SONLANDI (Kod: ${code}) ===\n`);
            trainingProcess = null;
            if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.send('training-finished', code || 0);
            }
        });
    } catch (err) {
        const str = `\n[HATA] ${err.message}\n`;
        appendLog(str);
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('training-log', str);
            mainWindow.webContents.send('training-finished', 1);
        }
        trainingProcess = null;
    }
});

ipcMain.on('stop-training', () => {
    if (trainingProcess) {
        try { trainingProcess.kill(); } catch (e) {}
        trainingProcess = null;
        appendLog("\n[!] Eğitim kullanıcı tarafından durduruldu.\n");
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('training-log', "\n[!] Eğitim kullanıcı tarafından durduruldu.\n");
        }
    }
});
