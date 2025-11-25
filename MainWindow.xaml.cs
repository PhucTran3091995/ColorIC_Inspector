using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using ColorIC_Inspector.Services;
using System;
using System.IO;
using System.Threading;
using System.Windows.Media;

namespace ColorIC_Inspector
{
    public partial class MainWindow : Window, INotifyPropertyChanged
    {
        private readonly CameraService _cameraService;
        private readonly YoloOnnxHelper? _yoloHelper;
        private readonly CancellationTokenSource _inferenceCts = new();
        private DispatcherTimer _inspectionTimer;
        private WriteableBitmap? _writeableBitmap;
        private string _currentPixelFormatKey = string.Empty; // ví dụ "Mono8" hoặc "Bgr24"
        private byte[]? _pixelBuffer;                // reuse to avoid allocations
        private int _lastStride = 0;
        private int _debugSavedFrames = 0;
        private BitmapSource? _latestFrame;
        private bool _isAnalyzing;

        // Lazy-loaded settings control
        private SettingTabs? _settingTabsControl;


        // Collections
        public ObservableCollection<LogEntry> Logs { get; set; } = new ObservableCollection<LogEntry>();
        public ObservableCollection<ICType> ICTypes { get; set; } = new ObservableCollection<ICType>();

        public MainWindow()
        {
            InitializeComponent();
            DataContext = this;

            _cameraService = new CameraService();
            _cameraService.FrameReceived += OnFrameReceived;
            _cameraService.StatusChanged += OnCameraStatusChanged;
            _cameraService.ErrorOccurred += OnCameraError;

            // Timer cho logic kiểm tra sản phẩm (giả lập)
            _inspectionTimer = new DispatcherTimer();
            _inspectionTimer.Interval = TimeSpan.FromMilliseconds(500); // 2 sản phẩm/giây
            _inspectionTimer.Tick += InspectionTimer_Tick;
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            // Tự động start camera khi mở app (tùy chọn)
            // btnStart_Click(null, null); 
        }

        private void Window_Closing(object sender, CancelEventArgs e)
        {
            _inferenceCts.Cancel();
            _yoloHelper?.Dispose();
            _cameraService.Dispose();
        }

        // --- CAMERA HANDLERS ---

        private void OnFrameReceived(BitmapSource bmp)
        {
            if (bmp == null) return;

            // Ensure bmp is frozen (safe to share between threads)
            if (!bmp.IsFrozen)
            {
                try { bmp.Freeze(); } catch { /* ignore if already frozen or fails */ }
            }

            _latestFrame = bmp;

            // Create a key to see if WriteableBitmap needs recreation
            string key = $"{bmp.PixelWidth}x{bmp.PixelHeight}_{bmp.Format}";

            Action updateUi = () =>
            {
                // Quick debug: control exists?
                if (imgCameraFeed == null)
                {
                    return;
                }

                // Check control visibility/size
                if (!imgCameraFeed.IsVisible || imgCameraFeed.Visibility != Visibility.Visible)
                {
                }
                if (imgCameraFeed.ActualWidth < 2 || imgCameraFeed.ActualHeight < 2)
                {
                }

                try
                {
                    // Create or re-create WriteableBitmap if necessary
                    if (_writeableBitmap == null || _currentPixelFormatKey != key)
                    {
                        _writeableBitmap = new WriteableBitmap(
                            bmp.PixelWidth,
                            bmp.PixelHeight,
                            bmp.DpiX,
                            bmp.DpiY,
                            bmp.Format,
                            bmp.Palette);

                        _currentPixelFormatKey = key;
                        // assign as image source once; future frames will only WritePixels
                        imgCameraFeed.Source = _writeableBitmap;

                        // reset pixel buffer
                        _pixelBuffer = null;
                        _lastStride = 0;
                    }

                    // Calculate stride and prepare pixel buffer (reuse if possible)
                    int stride = (bmp.PixelWidth * bmp.Format.BitsPerPixel + 7) / 8;
                    int needed = stride * bmp.PixelHeight;
                    if (_pixelBuffer == null || _pixelBuffer.Length != needed)
                    {
                        _pixelBuffer = new byte[needed];
                        _lastStride = stride;
                    }

                    // Copy pixels from BitmapSource into reusable buffer
                    bmp.CopyPixels(_pixelBuffer, stride, 0);

                    // Optional debug: check some stats on first N bytes (cheap)
                    if (_debugSavedFrames % 30 == 0) // log occasionally (kept logic but no output)
                    {
                        int checkLen = Math.Min(1000, _pixelBuffer.Length);
                        int min = 255, max = 0;
                        long sum = 0;
                        for (int i = 0; i < checkLen; i++)
                        {
                            int v = _pixelBuffer[i];
                            if (v < min) min = v;
                            if (v > max) max = v;
                            sum += v;
                        }
                        double mean = sum / (double)checkLen;
                        _ = mean; // intentionally unused, kept computation if needed later
                    }

                    // Write pixels into WriteableBitmap
                    var rect = new Int32Rect(0, 0, bmp.PixelWidth, bmp.PixelHeight);
                    _writeableBitmap!.WritePixels(rect, _pixelBuffer, stride, 0);

                    // Optionally save a couple debug PNGs to desktop for verification
                    if (_debugSavedFrames < 2)
                    {
                        SaveBitmapSourceToPng(_writeableBitmap, $"debug_frame_{_debugSavedFrames}.png");
                    }

                    _debugSavedFrames++;
                }
                catch (Exception)
                {
                    // Fallback: try assigning the BitmapSource directly
                    try { imgCameraFeed.Source = bmp; }
                    catch { }
                }
            };

            // Update on UI thread without blocking caller thread
            if (Dispatcher.CheckAccess())
                updateUi();
            else
                Dispatcher.BeginInvoke(updateUi, DispatcherPriority.Render);
        }

        private void SaveBitmapSourceToPng(BitmapSource bmpSrc, string fileName)
        {
            try
            {
                var encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(bmpSrc));
                string path = System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), fileName);
                using (var fs = System.IO.File.Open(path, System.IO.FileMode.Create))
                {
                    encoder.Save(fs);
                }
            }
            catch (Exception)
            {
                // swallow errors for debug saving
            }
        }



        private void OnCameraStatusChanged(string status)
        {
            Dispatcher.Invoke(() =>
            {
                // Hiển thị status lên Title hoặc StatusBar nếu có
                this.Title = $"ColorIC Inspector - {status}";
            });
        }

        private void OnCameraError(string error)
        {
            Dispatcher.Invoke(() =>
            {
                Logs.Insert(0, new LogEntry { Timestamp = DateTime.Now.ToString("HH:mm:ss"), Result = "ERR", ICName = error });
            });
        }

        // --- UI EVENTS ---

        private void btnStart_Click(object sender, RoutedEventArgs e)
        {
            var btn = sender as Button;

            if (!_cameraService.IsStreaming)
            {
                _cameraService.Start();
                _inspectionTimer.Start(); // Bắt đầu logic kiểm tra
                if (btn != null) btn.Content = "STOP INSPECTION";
            }
            else
            {
                _cameraService.Stop();
                _inspectionTimer.Stop();
                if (btn != null) btn.Content = "START INSPECTION";
            }
        }

        // --- BUSINESS LOGIC (Giữ nguyên logic cũ của bạn) ---

        private async void InspectionTimer_Tick(object sender, EventArgs e)
        {
            // Logic giả lập kiểm tra OK/NG
            if (_isAnalyzing) return;

            var frame = _latestFrame;
            if (frame == null)
            {
                {
                    UpdateStatus("WAITING", null);
                    return;
                }
                if (_yoloHelper?.Session == null)
                {
                    UpdateStatus("NO MODEL", null);
                    return;
                }

                try
                {
                    _isAnalyzing = true;
                    var result = await _yoloHelper.AnalyzeAsync(frame, _inferenceCts.Token);

                    if (!string.IsNullOrWhiteSpace(result.Error))
                    {
                        UpdateStatus("ERROR", false);
                        AppendLog("Inference", result.Error!, false);
                        return;
                    }

                    bool isOk = string.Equals(result.Verdict, "OK", StringComparison.OrdinalIgnoreCase);
                    UpdateStatus(result.Verdict, isOk);

                    string icName = (cbICTypes.SelectedItem as ICType)?.Name ?? "Unknown";
                    string detail = result.Detections.Count > 0 ? $"{result.Detections.Count} detection(s)" : "No detections";
                    AppendLog(icName, detail, isOk);
                }
                catch (OperationCanceledException)
                {
                }
                finally
                {
                    _isAnalyzing = false;
                }
            }
        }
        private void cbICTypes_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (cbICTypes.SelectedItem is ICType selected)
            {
                AppendLog("Selection", $"Selected {selected.Name}", null);
            }
        }
        private void btnClearLogs_Click(object sender, RoutedEventArgs e)
        {
            Logs.Clear();
        }

        // --- NAVIGATION ---
        private void btnNavMain_Click(object sender, RoutedEventArgs e)
        {
            viewMain.Visibility = Visibility.Visible;
            viewSetting.Visibility = Visibility.Collapsed;

            // update nav button visual state
            btnNavMain.Tag = "Active";
            btnNavSetting.Tag = null;
        }

        // NAV: Setting - lazy load SettingTabs into viewSetting
        private void btnNavSetting_Click(object sender, RoutedEventArgs e)
        {
            viewMain.Visibility = Visibility.Collapsed;
            viewSetting.Visibility = Visibility.Visible;

            // update nav button visual state
            btnNavSetting.Tag = "Active";
            btnNavMain.Tag = null;

            if (_settingTabsControl == null)
            {
                _settingTabsControl = new SettingTabs();
                // Clear any placeholder children and add the control
                viewSetting.Children.Clear();
                viewSetting.Children.Add(_settingTabsControl);
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;
        protected void OnPropertyChanged(string name) => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));

        private void SeedDefaultICTypes()
        {
            if (ICTypes.Count > 0) return;

            ICTypes.Add(new ICType { Name = "IC-74HC00", Count = 1, Color = "Black" });
            ICTypes.Add(new ICType { Name = "IC-LM358", Count = 2, Color = "DarkGrey" });
        }

        private YoloOnnxHelper? TryCreateYoloHelper()
        {
            var baseDir = AppDomain.CurrentDomain.BaseDirectory;
            var modelPath = Path.Combine(baseDir, "model.onnx");
            var yamlPath = Path.Combine(baseDir, "data.yaml");

            try
            {
                if (!File.Exists(modelPath) || !File.Exists(yamlPath))
                {
                    AppendLog("Model", "Missing model.onnx or data.yaml", false);
                    return null;
                }

                return new YoloOnnxHelper(modelPath, yamlPath);
            }
            catch (Exception ex)
            {
                AppendLog("Model", ex.Message, false);
                return null;
            }
        }

        private void UpdateStatus(string result, bool? isOk)
        {
            txtStatus.Text = result;
            if (!isOk.HasValue) return;

            try
            {
                txtStatus.Foreground = isOk.Value
                    ? (Brush)FindResource("Success")
                    : (Brush)FindResource("Error");
            }
            catch
            {
            }
        }

        private void AppendLog(string icName, string resultDetail, bool? isOk)
        {
            string verdict = isOk.HasValue ? (isOk.Value ? "OK" : "NG") : resultDetail;

            Logs.Insert(0, new LogEntry
            {
                Timestamp = DateTime.Now.ToString("HH:mm:ss"),
                ICName = string.IsNullOrWhiteSpace(resultDetail) ? icName : $"{icName} - {resultDetail}",
                Result = verdict
            });
        }
    }


    public class LogEntry
    {
        public string Timestamp { get; set; }
        public string ICName { get; set; }
        public string Result { get; set; }
    }

    public class ICType
    {
        public string Name { get; set; }
        public int Count { get; set; }
        public string Color { get; set; }
    }
}