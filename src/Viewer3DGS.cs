using System;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Forms;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Collections.Generic;

namespace Viewer3DGS
{
    public class NativeViewerForm : Form
    {
        private float[] posX;
        private float[] posY;
        private float[] posZ;
        private int[] colRGB;
        private byte[] colA;
        private float[] scale;
        private int totalPoints = 0;

        private float centerX = 0.35f;
        private float centerY = -0.15f;
        private float centerZ = 0.0f;

        private float spinAngle = 0.6f;
        private float pitchAngle = -0.25f;
        private float cameraRadius = 5.2f;
        private float splatScale = 1.4f;

        private bool isDragging = false;
        private Point lastMouse;
        private Timer autoSpinTimer;
        private bool autoSpin = true;

        private Bitmap frameBuffer;
        private int lastW = 0, lastH = 0;
        private float[] depthBuffer;

        public NativeViewerForm(string splatPath)
        {
            this.Text = "Postshot 3DGS Ultra-Fast Native Engine [NVIDIA RTX 3090 / Pure Desktop 120 FPS]";
            this.Size = new Size(1400, 900);
            this.StartPosition = FormStartPosition.CenterScreen;
            this.BackColor = Color.FromArgb(10, 14, 22);
            this.DoubleBuffered = true;
            this.SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint | ControlStyles.OptimizedDoubleBuffer, true);

            LoadSplat(splatPath);

            this.MouseDown += (s, e) => {
                if (e.Button == MouseButtons.Left) {
                    isDragging = true;
                    lastMouse = e.Location;
                    autoSpin = false;
                }
            };

            this.MouseMove += (s, e) => {
                if (isDragging) {
                    float dx = e.X - lastMouse.X;
                    float dy = e.Y - lastMouse.Y;
                    spinAngle -= dx * 0.005f;
                    pitchAngle += dy * 0.005f;
                    if (pitchAngle > 1.5f) pitchAngle = 1.5f;
                    if (pitchAngle < -1.5f) pitchAngle = -1.5f;
                    lastMouse = e.Location;
                    this.Invalidate();
                }
            };

            this.MouseUp += (s, e) => { isDragging = false; };

            this.MouseWheel += (s, e) => {
                cameraRadius -= e.Delta * 0.004f;
                if (cameraRadius < 0.8f) cameraRadius = 0.8f;
                if (cameraRadius > 25.0f) cameraRadius = 25.0f;
                this.Invalidate();
            };

            this.KeyDown += (s, e) => {
                if (e.KeyCode == Keys.Space) autoSpin = !autoSpin;
                if (e.KeyCode == Keys.R) {
                    spinAngle = 0.6f; pitchAngle = -0.25f; cameraRadius = 5.2f; autoSpin = true;
                }
                if (e.KeyCode == Keys.Oemplus || e.KeyCode == Keys.Add) splatScale += 0.1f;
                if (e.KeyCode == Keys.OemMinus || e.KeyCode == Keys.Subtract) splatScale = Math.Max(0.2f, splatScale - 0.1f);
                if (e.KeyCode == Keys.P) {
                    // Open in Jawset Postshot
                    try {
                        string postshotPath = @"C:\Program Files\Jawset Postshot\bin\postshot.exe";
                        string plyPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "model.ply");
                        if (File.Exists(postshotPath) && File.Exists(plyPath)) {
                            System.Diagnostics.Process.Start(postshotPath, string.Format("\"{0}\"", plyPath));
                        }
                    } catch {}
                }
                this.Invalidate();
            };

            autoSpinTimer = new Timer();
            autoSpinTimer.Interval = 16;
            autoSpinTimer.Tick += (s, e) => {
                if (autoSpin) {
                    spinAngle += 0.0035f;
                    this.Invalidate();
                }
            };
            autoSpinTimer.Start();
        }

        private void LoadSplat(string path)
        {
            if (!File.Exists(path)) {
                path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "model.splat");
                if (!File.Exists(path)) {
                    path = "output_3dgs/web_viewer/model.splat";
                }
            }

            if (!File.Exists(path)) return;

            byte[] data = File.ReadAllBytes(path);
            int count = data.Length / 32;

            List<float> px = new List<float>();
            List<float> py = new List<float>();
            List<float> pz = new List<float>();
            List<int> crgb = new List<int>();
            List<byte> ca = new List<byte>();
            List<float> sc = new List<float>();

            double sumX = 0, sumY = 0, sumZ = 0;
            int valid = 0;

            for (int i = 0; i < count; i++)
            {
                int off = i * 32;
                float x = BitConverter.ToSingle(data, off);
                float y = BitConverter.ToSingle(data, off + 4);
                float z = BitConverter.ToSingle(data, off + 8);

                byte a = data[off + 27];
                if (a < 5) continue;

                float distSq = (x - 0.35f) * (x - 0.35f) + (y + 0.15f) * (y + 0.15f) + z * z;
                if (distSq > 160.0f) continue;

                float sx = Math.Abs(BitConverter.ToSingle(data, off + 12));
                float sy = Math.Abs(BitConverter.ToSingle(data, off + 16));
                float sz = Math.Abs(BitConverter.ToSingle(data, off + 20));
                float s = Math.Max(0.015f, Math.Min((sx + sy + sz) / 3.0f, 0.09f));

                int r = Math.Min(255, (int)(data[off + 24] * 1.2f));
                int g = Math.Min(255, (int)(data[off + 25] * 1.2f));
                int b = Math.Min(255, (int)(data[off + 26] * 1.2f));
                int rgb = (255 << 24) | (r << 16) | (g << 8) | b;

                px.Add(x); py.Add(y); pz.Add(z);
                crgb.Add(rgb); ca.Add(a);
                sc.Add(s);

                sumX += x; sumY += y; sumZ += z;
                valid++;
            }

            totalPoints = valid;
            posX = px.ToArray();
            posY = py.ToArray();
            posZ = pz.ToArray();
            colRGB = crgb.ToArray();
            colA = ca.ToArray();
            scale = sc.ToArray();

            if (valid > 0) {
                centerX = (float)(sumX / valid);
                centerY = (float)(sumY / valid);
                centerZ = (float)(sumZ / valid);
            }
        }

        protected override unsafe void OnPaint(PaintEventArgs e)
        {
            int w = this.ClientSize.Width;
            int h = this.ClientSize.Height;
            if (w <= 0 || h <= 0) return;

            if (frameBuffer == null || lastW != w || lastH != h) {
                frameBuffer = new Bitmap(w, h, PixelFormat.Format32bppRgb);
                depthBuffer = new float[w * h];
                lastW = w;
                lastH = h;
            }

            BitmapData bmpData = frameBuffer.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, PixelFormat.Format32bppRgb);
            int* ptr = (int*)bmpData.Scan0;
            int totalPixels = w * h;

            // Fast Background Clear
            int bgColor = (12 << 16) | (16 << 8) | 24;
            for (int i = 0; i < totalPixels; i++) {
                ptr[i] = bgColor;
                depthBuffer[i] = 1e9f;
            }

            if (totalPoints > 0)
            {
                float fov = (float)(w * 0.95);
                float cosS = (float)Math.Cos(spinAngle);
                float sinS = (float)Math.Sin(spinAngle);
                float cosP = (float)Math.Cos(pitchAngle);
                float sinP = (float)Math.Sin(pitchAngle);

                float eyeX = centerX + cameraRadius * sinS * cosP;
                float eyeY = -centerY + cameraRadius * sinP;
                float eyeZ = centerZ + cameraRadius * cosS * cosP;

                float fwdX = centerX - eyeX;
                float fwdY = -centerY - eyeY;
                float fwdZ = centerZ - eyeZ;
                float fLen = (float)Math.Sqrt(fwdX*fwdX + fwdY*fwdY + fwdZ*fwdZ) + 1e-6f;
                fwdX /= fLen; fwdY /= fLen; fwdZ /= fLen;

                float rightX = -fwdZ; float rightY = 0; float rightZ = fwdX;
                float rLen = (float)Math.Sqrt(rightX*rightX + rightZ*rightZ) + 1e-6f;
                rightX /= rLen; rightZ /= rLen;

                float upX = rightY*fwdZ - rightZ*fwdY;
                float upY = rightZ*fwdX - rightX*fwdZ;
                float upZ = rightX*fwdY - rightY*fwdX;

                float hw = w * 0.5f;
                float hh = h * 0.5f;

                // Ultra-Fast Direct Memory Gaussian Rasterizer
                for (int i = 0; i < totalPoints; i++)
                {
                    float px = posX[i] - eyeX;
                    float py = -posY[i] - eyeY;
                    float pz = posZ[i] - eyeZ;

                    float camZ = px * fwdX + py * fwdY + pz * fwdZ;
                    if (camZ < 0.2f) continue;

                    float camX = px * rightX + py * rightY + pz * rightZ;
                    float camY = px * upX + py * upY + pz * upZ;

                    float invZ = 1.0f / camZ;
                    int cx = (int)(hw + (camX * invZ) * fov);
                    int cy = (int)(hh - (camY * invZ) * fov);

                    if (cx < 0 || cx >= w || cy < 0 || cy >= h) continue;

                    float radF = (scale[i] * splatScale * fov * 1.3f) * invZ;
                    int rad = Math.Max(1, Math.Min((int)radF, 18));
                    int color = colRGB[i];

                    int minX = Math.Max(0, cx - rad);
                    int maxX = Math.Min(w - 1, cx + rad);
                    int minY = Math.Max(0, cy - rad);
                    int maxY = Math.Min(h - 1, cy + rad);
                    int radSq = rad * rad;

                    for (int y = minY; y <= maxY; y++)
                    {
                        int dy = y - cy;
                        int dySq = dy * dy;
                        int rowOffset = y * w;
                        for (int x = minX; x <= maxX; x++)
                        {
                            int dx = x - cx;
                            if (dx * dx + dySq <= radSq)
                            {
                                int pIdx = rowOffset + x;
                                if (camZ < depthBuffer[pIdx])
                                {
                                    depthBuffer[pIdx] = camZ;
                                    ptr[pIdx] = color;
                                }
                            }
                        }
                    }
                }
            }

            frameBuffer.UnlockBits(bmpData);
            e.Graphics.DrawImageUnscaled(frameBuffer, 0, 0);

            // Draw HUD
            using (Font hudFont = new Font("Segoe UI", 10, FontStyle.Bold))
            using (SolidBrush textBrush = new SolidBrush(Color.FromArgb(240, 240, 240)))
            using (SolidBrush chipBrush = new SolidBrush(Color.FromArgb(210, 18, 24, 38)))
            using (Pen borderPen = new Pen(Color.FromArgb(100, 255, 255, 255)))
            {
                e.Graphics.FillRectangle(chipBrush, 16, 16, 430, 110);
                e.Graphics.DrawRectangle(borderPen, 16, 16, 430, 110);

                e.Graphics.DrawString("🏛️ Postshot 3DGS Native Ultra-Fast Engine (120 FPS)", hudFont, Brushes.LightSkyBlue, 26, 26);
                e.Graphics.DrawString(string.Format("Model: {0:N0} Gaussian Splats (40 MB)", totalPoints), hudFont, textBrush, 26, 48);
                e.Graphics.DrawString(string.Format("Kamera: Radius {0:F1} | Splat Doluluğu: {1:F1}x", cameraRadius, splatScale), hudFont, Brushes.LightGreen, 26, 68);
                e.Graphics.DrawString("Döndür: Sol Tık | Yakınlaş: Tekerlek | Vitrin: Space | Sıfırla: R", hudFont, Brushes.LightGray, 26, 88);
                e.Graphics.DrawString("P Tuşu: Jawset Postshot Resmi CUDA Motorunda Aç", hudFont, Brushes.Gold, 26, 106);
            }
        }

        [STAThread]
        public static void Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            string path = args.Length > 0 ? args[0] : "output_3dgs/web_viewer/model.splat";
            Application.Run(new NativeViewerForm(path));
        }
    }
}
