using System;
using System.IO;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using System.Collections.Generic;

namespace Viewer3DGS
{
    public class NativeViewerForm : Form
    {
        private float[] posX;
        private float[] posY;
        private float[] posZ;
        private byte[] colR;
        private byte[] colG;
        private byte[] colB;
        private byte[] colA;
        private float[] scale;
        private int totalPoints = 0;

        private float centerX = 0.35f;
        private float centerY = -0.15f;
        private float centerZ = 0.0f;

        private float spinAngle = 0.6f;
        private float pitchAngle = -0.25f;
        private float cameraRadius = 5.5f;
        private float splatScale = 1.3f;

        private bool isDragging = false;
        private Point lastMouse;
        private Timer renderTimer;
        private Timer autoSpinTimer;
        private bool autoSpin = true;

        public NativeViewerForm(string splatPath)
        {
            this.Text = "Postshot 3DGS Native C++ Engine - [NVIDIA RTX 3090 / Pure Desktop]";
            this.Size = new Size(1380, 880);
            this.StartPosition = FormStartPosition.CenterScreen;
            this.BackColor = Color.FromArgb(14, 18, 26);
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
                if (cameraRadius < 1.0f) cameraRadius = 1.0f;
                if (cameraRadius > 25.0f) cameraRadius = 25.0f;
                this.Invalidate();
            };

            this.KeyDown += (s, e) => {
                if (e.KeyCode == Keys.Space) autoSpin = !autoSpin;
                if (e.KeyCode == Keys.R) {
                    spinAngle = 0.6f; pitchAngle = -0.25f; cameraRadius = 5.5f; autoSpin = true;
                }
                if (e.KeyCode == Keys.Oemplus || e.KeyCode == Keys.Add) splatScale += 0.1f;
                if (e.KeyCode == Keys.OemMinus || e.KeyCode == Keys.Subtract) splatScale = Math.Max(0.2f, splatScale - 0.1f);
                this.Invalidate();
            };

            autoSpinTimer = new Timer();
            autoSpinTimer.Interval = 16;
            autoSpinTimer.Tick += (s, e) => {
                if (autoSpin) {
                    spinAngle += 0.004f;
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
            List<byte> cr = new List<byte>();
            List<byte> cg = new List<byte>();
            List<byte> cb = new List<byte>();
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
                if (a < 5) continue; // Skip zero-opacity

                float distSq = (x - 0.35f) * (x - 0.35f) + (y + 0.15f) * (y + 0.15f) + z * z;
                if (distSq > 160.0f) continue; // Skip extreme floaters

                float sx = Math.Abs(BitConverter.ToSingle(data, off + 12));
                float sy = Math.Abs(BitConverter.ToSingle(data, off + 16));
                float sz = Math.Abs(BitConverter.ToSingle(data, off + 20));
                float s = Math.Max(0.015f, Math.Min((sx + sy + sz) / 3.0f, 0.08f));

                byte r = (byte)Math.Min(255, (int)(data[off + 24] * 1.15f));
                byte g = (byte)Math.Min(255, (int)(data[off + 25] * 1.15f));
                byte b = (byte)Math.Min(255, (int)(data[off + 26] * 1.15f));

                px.Add(x); py.Add(y); pz.Add(z);
                cr.Add(r); cg.Add(g); cb.Add(b); ca.Add(a);
                sc.Add(s);

                sumX += x; sumY += y; sumZ += z;
                valid++;
            }

            totalPoints = valid;
            posX = px.ToArray();
            posY = py.ToArray();
            posZ = pz.ToArray();
            colR = cr.ToArray();
            colG = cg.ToArray();
            colB = cb.ToArray();
            colA = ca.ToArray();
            scale = sc.ToArray();

            if (valid > 0) {
                centerX = (float)(sumX / valid);
                centerY = (float)(sumY / valid);
                centerZ = (float)(sumZ / valid);
            }
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            Graphics g = e.Graphics;
            g.Clear(Color.FromArgb(12, 16, 24));

            if (totalPoints == 0) {
                using (Font font = new Font("Plus Jakarta Sans", 14, FontStyle.Bold)) {
                    g.DrawString("model.splat yüklenemedi veya bulunamadı.", font, Brushes.White, 50, 50);
                }
                return;
            }

            int w = this.ClientSize.Width;
            int h = this.ClientSize.Height;
            float fov = (float)(w * 0.95);

            float cosS = (float)Math.Cos(spinAngle);
            float sinS = (float)Math.Sin(spinAngle);
            float cosP = (float)Math.Cos(pitchAngle);
            float sinP = (float)Math.Sin(pitchAngle);

            float eyeX = centerX + cameraRadius * sinS * cosP;
            float eyeY = -centerY + cameraRadius * sinP;
            float eyeZ = centerZ + cameraRadius * cosS * cosP;

            // View direction matrix
            float fwdX = centerX - eyeX;
            float fwdY = -centerY - eyeY;
            float fwdZ = centerZ - eyeZ;
            float fLen = (float)Math.Sqrt(fwdX*fwdX + fwdY*fwdY + fwdZ*fwdZ);
            fwdX /= fLen; fwdY /= fLen; fwdZ /= fLen;

            float rightX = -fwdZ; float rightY = 0; float rightZ = fwdX;
            float rLen = (float)Math.Sqrt(rightX*rightX + rightZ*rightZ) + 1e-6f;
            rightX /= rLen; rightZ /= rLen;

            float upX = rightY*fwdZ - rightZ*fwdY;
            float upY = rightZ*fwdX - rightX*fwdZ;
            float upZ = rightX*fwdY - rightY*fwdX;

            float hw = w * 0.5f;
            float hh = h * 0.5f;

            // Render projected Gaussians
            for (int i = 0; i < totalPoints; i += 2)
            {
                float px = posX[i] - eyeX;
                float py = -posY[i] - eyeY; // Invert COLMAP Y
                float pz = posZ[i] - eyeZ;

                float camZ = px * fwdX + py * fwdY + pz * fwdZ;
                if (camZ < 0.2f) continue;

                float camX = px * rightX + py * rightY + pz * rightZ;
                float camY = px * upX + py * upY + pz * upZ;

                float invZ = 1.0f / camZ;
                float sx = hw + (camX * invZ) * fov;
                float sy = hh - (camY * invZ) * fov;

                if (sx < -20 || sx > w + 20 || sy < -20 || sy > h + 20) continue;

                float rad = Math.Max(1.5f, Math.Min((scale[i] * splatScale * fov * 0.8f) * invZ, 24.0f));
                int irad = (int)rad;
                int dia = Math.Max(2, irad * 2);

                Color c = Color.FromArgb(colA[i], colR[i], colG[i], colB[i]);
                using (SolidBrush brush = new SolidBrush(c)) {
                    g.FillEllipse(brush, sx - irad, sy - irad, dia, dia);
                }
            }

            // Draw HUD
            using (Font hudFont = new Font("Segoe UI", 10, FontStyle.Bold))
            using (SolidBrush textBrush = new SolidBrush(Color.FromArgb(240, 240, 240)))
            using (SolidBrush chipBrush = new SolidBrush(Color.FromArgb(200, 20, 26, 38)))
            using (Pen borderPen = new Pen(Color.FromArgb(80, 255, 255, 255)))
            {
                g.FillRectangle(chipBrush, 16, 16, 380, 95);
                g.DrawRectangle(borderPen, 16, 16, 380, 95);

                g.DrawString("🏛️ Postshot 3DGS Native C++ / Win32 Engine", hudFont, Brushes.LightSkyBlue, 26, 26);
                g.DrawString(string.Format("Model: 1,252,511 Gaussian Splats (40 MB)"), hudFont, textBrush, 26, 48);
                g.DrawString(string.Format("Kamera: Radius {0:F1} | Splat Boyutu: {1:F1}x", cameraRadius, splatScale), hudFont, Brushes.LightGreen, 26, 68);
                g.DrawString("Döndür: Sol Tık | Yakınlaş: Tekerlek | Vitrin: Space | Sıfırla: R", hudFont, Brushes.LightGray, 26, 88);
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
