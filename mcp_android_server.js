#!/usr/bin/env node
/**
 * Android Studio & ADB Model Context Protocol (MCP) Server
 * Stdio JSON-RPC 2.0 server for Antigravity & AI agents
 */

const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

// Locate ADB executable
const defaultAdbPath = "C:\\Users\\halit\\AppData\\Local\\Android\\Sdk\\platform-tools\\adb.exe";
let adbPath = fs.existsSync(defaultAdbPath) ? defaultAdbPath : "adb";

function runAdb(args) {
    try {
        const cmd = `"${adbPath}" ${args}`;
        return execSync(cmd, { encoding: 'utf-8' }).trim();
    } catch (e) {
        return `Error executing adb ${args}: ${e.message}`;
    }
}

const TOOLS = [
    {
        name: "adb_list_devices",
        description: "List all connected Android physical devices and running emulators.",
        inputSchema: {
            type: "object",
            properties: {}
        }
    },
    {
        name: "adb_reverse_port",
        description: "Reverse forward a TCP port from the Android phone to PC (e.g. phone localhost:8080 -> PC localhost:8080).",
        inputSchema: {
            type: "object",
            properties: {
                device_port: { type: "number", description: "Port on the Android device (e.g. 8080)", default: 8080 },
                host_port: { type: "number", description: "Port on the host PC (e.g. 8080)", default: 8080 }
            },
            required: ["device_port", "host_port"]
        }
    },
    {
        name: "adb_launch_url",
        description: "Launch a URL in the default browser or Chrome on the connected Android phone.",
        inputSchema: {
            type: "object",
            properties: {
                url: { type: "string", description: "URL to open on phone (e.g. http://localhost:8080)" }
            },
            required: ["url"]
        }
    },
    {
        name: "adb_screenshot",
        description: "Capture a live screenshot of the phone screen and save it to the specified image path.",
        inputSchema: {
            type: "object",
            properties: {
                output_path: { type: "string", description: "Absolute path where the PNG screenshot should be saved." }
            },
            required: ["output_path"]
        }
    },
    {
        name: "adb_device_info",
        description: "Get Android phone device model, brand, Android version, screen resolution, and battery status.",
        inputSchema: {
            type: "object",
            properties: {}
        }
    },
    {
        name: "android_studio_status",
        description: "Check Android Studio installation status, SDK location, and project availability.",
        inputSchema: {
            type: "object",
            properties: {}
        }
    }
];

function handleToolCall(name, args) {
    switch (name) {
        case "adb_list_devices": {
            const out = runAdb("devices -l");
            return { content: [{ type: "text", text: out }] };
        }
        case "adb_reverse_port": {
            const devPort = args.device_port || 8080;
            const hostPort = args.host_port || 8080;
            const res = runAdb(`reverse tcp:${devPort} tcp:${hostPort}`);
            return { content: [{ type: "text", text: `Reverse port forwarding established: device tcp:${devPort} -> host tcp:${hostPort}. Result: ${res || 'OK'}` }] };
        }
        case "adb_launch_url": {
            const url = args.url || "http://localhost:8080";
            const res = runAdb(`shell am start -a android.intent.action.VIEW -d "${url}"`);
            return { content: [{ type: "text", text: `Launched URL "${url}" on Android device.\n${res}` }] };
        }
        case "adb_screenshot": {
            const savePath = args.output_path || path.join(__dirname, "phone_screenshot.png");
            try {
                execSync(`"${adbPath}" exec-out screencap -p > "${savePath}"`, { shell: 'powershell.exe' });
                return { content: [{ type: "text", text: `Screenshot captured successfully to: ${savePath}` }] };
            } catch (e) {
                return { content: [{ type: "text", text: `Failed to capture screenshot: ${e.message}` }] };
            }
        }
        case "adb_device_info": {
            const model = runAdb("shell getprop ro.product.model");
            const brand = runAdb("shell getprop ro.product.brand");
            const osVer = runAdb("shell getprop ro.build.version.release");
            const sdkVer = runAdb("shell getprop ro.build.version.sdk");
            const wmSize = runAdb("shell wm size");
            return {
                content: [{
                    type: "text",
                    text: `Android Device Info:\n- Brand: ${brand}\n- Model: ${model}\n- Android OS: ${osVer} (SDK ${sdkVer})\n- Resolution: ${wmSize}`
                }]
            };
        }
        case "android_studio_status": {
            const studioPath = "C:\\Program Files\\Android\\Android Studio";
            const sdkPath = "C:\\Users\\halit\\AppData\\Local\\Android\\Sdk";
            return {
                content: [{
                    type: "text",
                    text: `Android Studio Environment:\n- Android Studio: ${fs.existsSync(studioPath) ? 'Installed at ' + studioPath : 'Not found'}\n- Android SDK: ${fs.existsSync(sdkPath) ? 'Found at ' + sdkPath : 'Not found'}\n- ADB Path: ${adbPath}`
                }]
            };
        }
        default:
            throw new Error(`Unknown tool: ${name}`);
    }
}

// JSON-RPC stdio handler
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
});

rl.on('line', (line) => {
    if (!line.trim()) return;
    try {
        const req = JSON.parse(line);
        const { id, method, params } = req;

        if (method === "initialize") {
            const response = {
                jsonrpc: "2.0",
                id,
                result: {
                    protocolVersion: "2024-11-05",
                    capabilities: { tools: {} },
                    serverInfo: {
                        name: "android-studio-mcp-server",
                        version: "1.0.0"
                    }
                }
            };
            process.stdout.write(JSON.stringify(response) + "\n");
        } else if (method === "notifications/initialized") {
            // Notification, no response needed
        } else if (method === "tools/list") {
            const response = {
                jsonrpc: "2.0",
                id,
                result: { tools: TOOLS }
            };
            process.stdout.write(JSON.stringify(response) + "\n");
        } else if (method === "tools/call") {
            try {
                const toolResult = handleToolCall(params.name, params.arguments || {});
                const response = {
                    jsonrpc: "2.0",
                    id,
                    result: toolResult
                };
                process.stdout.write(JSON.stringify(response) + "\n");
            } catch (err) {
                const response = {
                    jsonrpc: "2.0",
                    id,
                    error: { code: -32603, message: err.message }
                };
                process.stdout.write(JSON.stringify(response) + "\n");
            }
        } else {
            const response = {
                jsonrpc: "2.0",
                id,
                error: { code: -32601, message: `Method not found: ${method}` }
            };
            process.stdout.write(JSON.stringify(response) + "\n");
        }
    } catch (e) {
        console.error("JSON parse error", e);
    }
});
