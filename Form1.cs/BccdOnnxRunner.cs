// NuGet: Microsoft.ML.OnnxRuntime
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;

public sealed class BccdOnnxRunner : IDisposable
{
    private readonly InferenceSession _sess;
    private readonly string _inputName;
    private readonly string[] _classes = { "WBC", "RBC", "Platelets" }; // 학습 클래스 순서

    // mmdetection normalize(mean/std, to_rgb=True)
    private static readonly float[] Mean = { 123.675f, 116.28f, 103.53f };
    private static readonly float[] Std = { 58.395f, 57.12f, 57.375f };

    public BccdOnnxRunner(string onnxPath)
    {
        _sess = new InferenceSession(onnxPath);
        _inputName = _sess.InputMetadata.Keys.First(); // 일반적으로 "input"
    }

    public void Dispose()
    {
        _sess.Dispose();
    }

    private static Rectangle ComputeLetterbox(Size src, int maxW, int maxH, out float scale)
    {
        float r = Math.Min((float)maxW / src.Width, (float)maxH / src.Height);
        scale = r;
        int newW = (int)Math.Round(src.Width * r);
        int newH = (int)Math.Round(src.Height * r);
        return new Rectangle(0, 0, newW, newH);
    }

    private static Size PadToDiv(Size s, int div)
    {
        int padW = (int)Math.Ceiling(s.Width / (float)div) * div;
        int padH = (int)Math.Ceiling(s.Height / (float)div) * div;
        return new Size(padW, padH);
    }

    private static Bitmap ResizeKeepRatioAndPad(
        Bitmap src, int maxW, int maxH, int divisor,
        out float scale, out Size resized, out Size padded)
    {
        Rectangle rect = ComputeLetterbox(src.Size, maxW, maxH, out scale);
        resized = rect.Size;

        Bitmap tmp = new Bitmap(rect.Width, rect.Height, PixelFormat.Format24bppRgb);
        using (Graphics g = Graphics.FromImage(tmp))
        {
            g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            g.DrawImage(src, new Rectangle(0, 0, rect.Width, rect.Height));
        }

        Size padSize = PadToDiv(tmp.Size, divisor); // size_divisor=32
        padded = padSize;
        if (padSize == tmp.Size)
            return tmp;

        Bitmap canvas = new Bitmap(padSize.Width, padSize.Height, PixelFormat.Format24bppRgb);
        using (Graphics g = Graphics.FromImage(canvas))
        {
            g.Clear(Color.Black); // 패딩 0 → Normalize 후 분포 유지
            g.DrawImage(tmp, 0, 0);
        }
        tmp.Dispose();
        return canvas;
    }

    private static float[] ToCHWAndNormalizeFast(Bitmap b)
    {
        int W = b.Width, H = b.Height;
        float[] data = new float[1 * 3 * H * W];

        var rect = new Rectangle(0, 0, W, H);
        var bmpData = b.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
        try
        {
            unsafe
            {
                byte* src = (byte*)bmpData.Scan0;
                int stride = bmpData.Stride;
                for (int y = 0; y < H; y++)
                {
                    byte* row = src + y * stride;
                    for (int x = 0; x < W; x++)
                    {
                        // BGR 메모리 순서
                        byte b0 = row[x * 3 + 0];
                        byte g0 = row[x * 3 + 1];
                        byte r0 = row[x * 3 + 2];

                        int idx = y * W + x;
                        data[0 * H * W + idx] = (r0 - Mean[0]) / Std[0];
                        data[1 * H * W + idx] = (g0 - Mean[1]) / Std[1];
                        data[2 * H * W + idx] = (b0 - Mean[2]) / Std[2];
                    }
                }
            }
        }
        finally { b.UnlockBits(bmpData); }
        return data;
    }


    public Tuple<float[][], int[]> Infer(
    Bitmap src, float scoreThr = 0.30f, int maxW = 1333, int maxH = 800, int divisor = 32)
    {
        // 1) Resize keep_ratio → Pad(32) → Normalize(+CHW)
        float scale;
        Size resized, padded;
        Bitmap pre = ResizeKeepRatioAndPad(src, maxW, maxH, divisor, out scale, out resized, out padded);
        try
        {
            float[] chw = ToCHWAndNormalizeFast(pre);

            // 2) Run
            var inputT = new DenseTensor<float>(chw, new[] { 1, 3, padded.Height, padded.Width });
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, inputT) };

            IDisposable resultsDisposable = null;
            try
            {
                var results = _sess.Run(inputs);
                resultsDisposable = results;

                string detsName = results.First(x => x.Name.ToLower().Contains("det")).Name;
                string labelsName = results.First(x => x.Name.ToLower().Contains("label")).Name;

                float[] detsTensor = results.First(r => r.Name == detsName).AsEnumerable<float>().ToArray();
                long[] labelsTensor = results.First(r => r.Name == labelsName).AsEnumerable<long>().ToArray();

                int N = detsTensor.Length / 5;
                List<float[]> dets = new List<float[]>(N);
                List<int> labels = new List<int>(N);

                // ★★★ 역보정: 리사이즈 스케일을 되돌려 원본 좌표로 변환 + 클램프
                float inv = (scale > 0f) ? (1f / scale) : 1f;
                int W0 = src.Width, H0 = src.Height;

                for (int i = 0; i < N; i++)
                {
                    float x1 = detsTensor[i * 5 + 0];
                    float y1 = detsTensor[i * 5 + 1];
                    float x2 = detsTensor[i * 5 + 2];
                    float y2 = detsTensor[i * 5 + 3];
                    float s = detsTensor[i * 5 + 4];
                    if (s < scoreThr) continue;

                    // 패딩은 오른쪽/아래쪽만 주었으므로 (0,0) 오프셋 없음 → 스케일만 되돌리면 됨
                    x1 *= inv; y1 *= inv; x2 *= inv; y2 *= inv;

                    // 클램프(혹시 패딩 영역 예측이 섞였을 경우를 대비)
                    if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
                    if (x2 > W0 - 1) x2 = W0 - 1;
                    if (y2 > H0 - 1) y2 = H0 - 1;

                    // 너무 작은 박스 제거(옵션)
                    if (x2 - x1 < 1 || y2 - y1 < 1) continue;

                    dets.Add(new float[] { x1, y1, x2, y2, s });
                    labels.Add((int)labelsTensor[i]);
                }

                return new Tuple<float[][], int[]>(dets.ToArray(), labels.ToArray());
            }
            finally
            {
                if (resultsDisposable != null) resultsDisposable.Dispose();
            }
        }
        finally
        {
            pre.Dispose();
        }
    }


    public void DrawOverlay(
        Graphics g, Tuple<float[][], int[]> pred, float scoreThr, Size canvasSize)
    {
        using (var penWBC = new Pen(Color.Lime, 2f))
        using (var penRBC = new Pen(Color.Red, 2f))
        using (var penPLT = new Pen(Color.Blue, 2f))
        using (var font = new Font("Segoe UI", 9f, FontStyle.Bold))
        using (var white = new SolidBrush(Color.White))
        {
            float[][] dets = pred.Item1;
            int[] labels = pred.Item2;

            for (int i = 0; i < dets.Length; i++)
            {
                float[] d = dets[i];
                if (d[4] < scoreThr) continue;

                int clsId = (i < labels.Length) ? labels[i] : 0;
                Pen pen;
                if (clsId == 0) pen = penWBC;
                else if (clsId == 1) pen = penRBC;
                else if (clsId == 2) pen = penPLT;
                else pen = penWBC;

                RectangleF rect = new RectangleF(d[0], d[1], d[2] - d[0], d[3] - d[1]);
                g.DrawRectangle(pen, rect.X, rect.Y, rect.Width, rect.Height);

                string clsName = _classes[Math.Min(clsId, _classes.Length - 1)];
                string label = string.Format("{0} {1:0.00}", clsName, d[4]);

                SizeF sz = g.MeasureString(label, font);
                using (var bg = new SolidBrush(Color.FromArgb(180, pen.Color)))
                {
                    g.FillRectangle(bg, rect.X, rect.Y - sz.Height, sz.Width, sz.Height);
                }
                g.DrawString(label, font, white, rect.X, rect.Y - sz.Height);
            }
        }
    }
}
