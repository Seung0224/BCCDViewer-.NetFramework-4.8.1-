using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Form1.cs
{
    public partial class Form1 : Form
    {
        private PictureBox _pic;
        private DataGridView _grid;
        private Button _btnOpen, _btnRun, _btnClear;
        private Label _lblInfo;

        private BccdOnnxRunner _runner;
        private string _onnxPath = @"D:\TOYPROJECT\BCCDViwer\mmdetection\deploy_onnx\end2end.onnx";
        private float _scoreThr = 0.30f;

        public Form1()
        {
            InitializeComponent();
            Text = "BCCD ONNX Viewer (.NET Framework 4.8.1)";
            Width = 1150;
            Height = 720;

            // 상단 툴바
            var top = new FlowLayoutPanel { Dock = DockStyle.Top, Height = 44, Padding = new Padding(8) };
            _btnOpen = new Button { Text = "이미지 열기", Width = 110, Height = 30 };
            _btnRun = new Button { Text = "추론 실행", Width = 110, Height = 30 };
            _btnClear = new Button { Text = "초기화", Width = 80, Height = 30 };
            _lblInfo = new Label { AutoSize = true, Text = "ONNX: " + _onnxPath };
            top.Controls.AddRange(new Control[] { _btnOpen, _btnRun, _btnClear, _lblInfo });
            Controls.Add(top);

            // 메인 뷰
            _pic = new PictureBox { Dock = DockStyle.Fill, BackColor = Color.Black, SizeMode = PictureBoxSizeMode.Zoom };
            _grid = new DataGridView { Dock = DockStyle.Right, Width = 380, ReadOnly = true, AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill };
            Controls.Add(_pic);
            Controls.Add(_grid);

            // 러너 초기화
            try
            {
                _runner = new BccdOnnxRunner(_onnxPath);
            }
            catch (Exception ex)
            {
                MessageBox.Show("ONNX 로드 실패: " + ex.Message);
            }

            // 이벤트
            _btnOpen.Click += (s, e) => OnOpen();
            _btnRun.Click += (s, e) => OnRun();
            _btnClear.Click += (s, e) => { _pic.Image = null; _grid.DataSource = null; };
        }
        private void OnOpen()
        {
            using (var ofd = new OpenFileDialog { Filter = "Images|*.jpg;*.png;*.bmp" })
            {
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    LoadImageNoLock(ofd.FileName);
                }
            }
        }

        private void OnRun()
        {
            if (_runner == null)
            {
                MessageBox.Show("ONNX 세션이 초기화되지 않았습니다.");
                return;
            }
            if (_pic.Image == null)
            {
                MessageBox.Show("이미지를 먼저 여세요.");
                return;
            }

            try
            {
                using (var src = new Bitmap(_pic.Image))
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();

                    var pred = _runner.Infer(src, _scoreThr); // 리사이즈/패딩+Normalize 포함

                    sw.Stop();
                    System.Diagnostics.Trace.WriteLine($"[ONNX] 추론 시간: {sw.Elapsed.TotalMilliseconds:0.00} ms");

                    // 오버레이
                    using (var vis = new Bitmap(src))
                    using (var g = Graphics.FromImage(vis))
                    {
                        _runner.DrawOverlay(g, pred, _scoreThr, vis.Size);
                        _pic.Image?.Dispose();
                        _pic.Image = new Bitmap(vis);
                    }

                    // 테이블 표시
                    var rows = pred.Item1.Select((d, i) => new
                    {
                        idx = i,
                        cls_id = (i < pred.Item2.Length) ? pred.Item2[i] : 0,
                        x1 = d[0],
                        y1 = d[1],
                        x2 = d[2],
                        y2 = d[3],
                        score = d[4]
                    }).ToList();
                    _grid.DataSource = rows;
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("추론 실패: " + ex.Message);
            }
        }


        // 파일 잠김 방지 미리보기 로더
        private void LoadImageNoLock(string path)
        {
            using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (var ms = new MemoryStream())
            {
                fs.CopyTo(ms);
                ms.Position = 0;
                _pic.Image?.Dispose();
                _pic.Image = new Bitmap(ms);
            }
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            if (_runner != null) _runner.Dispose();
            base.OnFormClosed(e);
        }
    }
}
