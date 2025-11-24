using Basler.Pylon;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ColorIC_Inspector
{
    public class CameraService : IDisposable
    {
        private Camera? _camera;
        private readonly PixelDataConverter _converter = new PixelDataConverter();
        private CancellationTokenSource? _cts;
        private Task? _grabTask;
        private readonly SemaphoreSlim _retrieveLock = new SemaphoreSlim(1, 1); // ensure single retriever

        // Events for UI
        public event Action<BitmapSource>? FrameReceived;
        public event Action<string>? StatusChanged;
        public event Action<string>? ErrorOccurred;

        public bool IsStreaming { get; private set; }
        public bool IsMocking { get; private set; }

        public void Start()
        {
            if (IsStreaming) return;
            IsStreaming = true;
            _cts = new CancellationTokenSource();

            _grabTask = Task.Run(() =>
            {
                try
                {
                    // Try real camera; if fails, fall back to mock
                    InitializeRealCamera(_cts.Token);
                }
                catch (OperationCanceledException)
                {
                    // expected on stop
                    Debug.WriteLine("InitializeRealCamera canceled.");
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Camera init failed: {ex.Message}. Switching to Mock mode.");
                    StatusChanged?.Invoke("Camera not found. Switching to Simulation Mode.");
                    ErrorOccurred?.Invoke($"Camera Error: {ex.Message}");
                    RunMockCamera(_cts.Token);
                }
                finally
                {
                    IsStreaming = false;
                }
            });
        }

        public void Stop()
        {
            try
            {
                _cts?.Cancel();

                // Wait a short time for grab loop to stop gracefully
                try
                {
                    _grabTask?.Wait(2000); // optional, avoid indefinite block
                }
                catch (AggregateException) { /* swallow */ }

                IsStreaming = false;
                IsMocking = false;
                StatusChanged?.Invoke("Stopped.");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Stop error: {ex}");
            }
            finally
            {
                // Ensure cleanup even if task didn't finish
                CleanupCamera();
            }
        }

        private void InitializeRealCamera(CancellationToken token)
        {
            try
            {
                _camera = new Camera();

                _camera.ConnectionLost += (s, e) =>
                {
                    ErrorOccurred?.Invoke("Connection lost!");
                    Stop();
                };

                _camera.Open();

                // --- TỰ ĐỘNG thử bật AutoExposure/AutoGain (giúp debug ảnh tối) ---
                try
                {
                    if (_camera.Parameters.Contains(PLCamera.ExposureAuto))
                        _camera.Parameters[PLCamera.ExposureAuto].TrySetValue(PLCamera.ExposureAuto.Continuous);

                    if (_camera.Parameters.Contains(PLCamera.GainAuto))
                        _camera.Parameters[PLCamera.GainAuto].TrySetValue(PLCamera.GainAuto.Continuous);

                    Debug.WriteLine("AutoExposure & AutoGain enabled for debug (if supported).");
                }
                catch (Exception ex)
                {
                    Debug.WriteLine("AutoExposure/Gain enable failed: " + ex.Message);
                }

                // --- CHỌN pixel format an toàn (ưu tiên Mono8, sau đó BayerRG8, else dùng mặc định) ---
                string chosenPixelFormat = string.Empty;
                try
                {
                    if (_camera.Parameters.Contains(PLCamera.PixelFormat))
                    {
                        // đọc hiện tại (để log)
                        chosenPixelFormat = _camera.Parameters[PLCamera.PixelFormat].GetValue();

                        if (_camera.Parameters[PLCamera.PixelFormat].CanSetValue(PLCamera.PixelFormat.Mono8))
                        {
                            _camera.Parameters[PLCamera.PixelFormat].SetValue(PLCamera.PixelFormat.Mono8);
                            chosenPixelFormat = PLCamera.PixelFormat.Mono8.ToString();
                            Debug.WriteLine("Using PixelFormat: Mono8");
                        }
                        else if (_camera.Parameters[PLCamera.PixelFormat].CanSetValue(PLCamera.PixelFormat.BayerRG8))
                        {
                            _camera.Parameters[PLCamera.PixelFormat].SetValue(PLCamera.PixelFormat.BayerRG8);
                            chosenPixelFormat = PLCamera.PixelFormat.BayerRG8.ToString();
                            Debug.WriteLine("Using PixelFormat: BayerRG8");
                        }
                        else
                        {
                            chosenPixelFormat = _camera.Parameters[PLCamera.PixelFormat].GetValue();
                            Debug.WriteLine($"Using camera default PixelFormat: {chosenPixelFormat}");
                        }
                    }
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error selecting PixelFormat: {ex}");
                    try
                    {
                        if (_camera.Parameters.Contains(PLCamera.PixelFormat))
                            chosenPixelFormat = _camera.Parameters[PLCamera.PixelFormat].GetValue();
                    }
                    catch { }
                }

                StatusChanged?.Invoke($"Connected: {_camera.CameraInfo[CameraInfoKey.ModelName]} (PixelFormat: {chosenPixelFormat})");

                // --- Map PixelFormat -> PixelDataConverter.OutputPixelFormat & WPF PixelFormat ---
                PixelType converterOutput = PixelType.BGRA8packed;
                System.Windows.Media.PixelFormat wpfPixelFormat = PixelFormats.Bgra32;
                int bytesPerPixel = 4;

                var pfLower = (chosenPixelFormat ?? string.Empty).ToLowerInvariant();
                if (pfLower.Contains("mono8"))
                {
                    converterOutput = PixelType.Mono8;
                    wpfPixelFormat = PixelFormats.Gray8;
                    bytesPerPixel = 1;
                }
                else if (pfLower.Contains("bayer") && pfLower.Contains("rg8"))
                {
                    converterOutput = PixelType.BGR8packed; // convert Bayer -> BGR
                    wpfPixelFormat = PixelFormats.Bgr24;
                    bytesPerPixel = 3;
                }
                else if (pfLower.Contains("rgb8") || pfLower.Contains("bgr8"))
                {
                    converterOutput = PixelType.BGR8packed;
                    wpfPixelFormat = PixelFormats.Bgr24;
                    bytesPerPixel = 3;
                }
                else if (pfLower.Contains("bgra8") || pfLower.Contains("rgba8"))
                {
                    converterOutput = PixelType.BGRA8packed;
                    wpfPixelFormat = PixelFormats.Bgra32;
                    bytesPerPixel = 4;
                }
                else
                {
                    // fallback
                    converterOutput = PixelType.BGRA8packed;
                    wpfPixelFormat = PixelFormats.Bgra32;
                    bytesPerPixel = 4;
                    Debug.WriteLine($"Unknown PixelFormat '{chosenPixelFormat}', falling back to BGRA8 for display.");
                }

                _converter.OutputPixelFormat = converterOutput;

                // --- Start grabbing (we retrieve ourselves) ---
                _camera.StreamGrabber.Start(GrabStrategy.LatestImages, GrabLoop.ProvidedByUser);

                byte[]? buffer = null;

                while (!token.IsCancellationRequested && _camera.StreamGrabber.IsGrabbing)
                {
                    bool entered = false;
                    try
                    {
                        // only one retriever at a time
                        _retrieveLock.Wait(token);
                        entered = true;

                        IGrabResult? result = null;
                        try
                        {
                            result = _camera.StreamGrabber.RetrieveResult(5000, TimeoutHandling.Return);

                            if (result == null)
                            {
                                // Timeout, continue loop
                                continue;
                            }

                            using (result)
                            {
                                if (!result.GrabSucceeded)
                                {
                                    Debug.WriteLine($"Grab failed: {result.ErrorCode} {result.ErrorDescription}");
                                    continue;
                                }

                                // get dimensions
                                int width = (int)result.Width;
                                int height = (int)result.Height;

                                // allocate buffer based on expected **converted** size (width*height*bytesPerPixel)
                                int expectedSize = width * height * bytesPerPixel;
                                if (buffer == null || buffer.Length != expectedSize)
                                    buffer = new byte[expectedSize];

                                // Debug info
                                Debug.WriteLine($"Width: {width}, Height: {height}, PayloadSize: {result.PayloadSize}, PixelType: {result.PixelTypeValue}, ExpectedConvertedSize: {expectedSize}");

                                // Convert into managed buffer (may throw SEHException if unsupported)
                                try
                                {
                                    _converter.Convert(buffer, result);
                                }
                                catch (System.Runtime.InteropServices.SEHException seh)
                                {
                                    Debug.WriteLine($"SEHException in Convert: {seh}");
                                    ErrorOccurred?.Invoke($"Convert error (native): {seh.Message}");
                                    // rethrow to allow fallback behavior upstream
                                    throw;
                                }
                                catch (Exception ex)
                                {
                                    Debug.WriteLine($"Convert exception: {ex}");
                                    ErrorOccurred?.Invoke($"Convert error: {ex.Message}");
                                    // continue to next frame
                                    continue;
                                }

                                // --- Quick buffer stats (debug) ---
                                int lenCheck = Math.Min(1000, buffer.Length);
                                int min = 255, max = 0;
                                long sum = 0;
                                for (int i = 0; i < lenCheck; i++)
                                {
                                    int v = buffer[i];
                                    if (v < min) min = v;
                                    if (v > max) max = v;
                                    sum += v;
                                }
                                double mean = sum / (double)lenCheck;
                                Debug.WriteLine($"Buffer stats (first {lenCheck} bytes): min={min}, max={max}, mean={mean:F2}");

                                if (max == 0)
                                {
                                    // All zeros -> likely exposure/lighting issue
                                    Debug.WriteLine("Buffer appears all zeros. Check exposure/lighting or enable AutoExposure.");
                                    ErrorOccurred?.Invoke("Frame appears empty (all zeros). Check exposure/lighting.");
                                }

                                // create BitmapSource with proper stride
                                int stride = width * bytesPerPixel;
                                var bmp = BitmapSource.Create(
                                    width, height,
                                    96, 96,
                                    wpfPixelFormat,
                                    null,
                                    buffer,
                                    stride);

                                bmp.Freeze();
                                FrameReceived?.Invoke(bmp);
                            }
                        }
                        finally
                        {
                            // nothing additional here
                        }
                    }
                    catch (OperationCanceledException)
                    {
                        Debug.WriteLine("Grab loop canceled.");
                        throw;
                    }
                    catch (System.Runtime.InteropServices.SEHException)
                    {
                        // serious native error — rethrow to fallback to mock
                        throw;
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine($"Retrieve error: {ex}");
                        ErrorOccurred?.Invoke($"Retrieve error: {ex.Message}");
                        // short delay to avoid busy loop on persistent errors
                        try { Task.Delay(50, CancellationToken.None).Wait(); } catch { }
                    }
                    finally
                    {
                        if (entered)
                        {
                            try { _retrieveLock.Release(); } catch { }
                        }
                    }
                }
            }
            finally
            {
                CleanupCamera();
            }
        }



        private void RunMockCamera(CancellationToken token)
        {
            IsMocking = true;
            StatusChanged?.Invoke("Running Simulation (No Camera)");

            var rnd = new Random();
            int width = 640;
            int height = 480;
            byte[] mockBuffer = new byte[width * height * 4];

            while (!token.IsCancellationRequested)
            {
                Thread.Sleep(33);
                rnd.NextBytes(mockBuffer);

                var bmp = BitmapSource.Create(
                    width, height,
                    96, 96,
                    PixelFormats.Bgra32,
                    null,
                    mockBuffer,
                    width * 4);

                bmp.Freeze();
                FrameReceived?.Invoke(bmp);
            }
        }

        private void CleanupCamera()
        {
            if (_camera != null)
            {
                try
                {
                    // Stop grabbing first
                    try
                    {
                        if (_camera.StreamGrabber.IsGrabbing)
                        {
                            _camera.StreamGrabber.Stop();
                        }
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine($"Error stopping StreamGrabber: {ex}");
                    }

                    try
                    {
                        if (_camera.IsOpen) _camera.Close();
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine($"Error closing camera: {ex}");
                    }

                    try { _camera.Dispose(); } catch { }
                }
                finally
                {
                    _camera = null;
                }
            }
        }

        public void Dispose()
        {
            Stop();
            _retrieveLock.Dispose();
            _cts?.Dispose();
        }
    }
}
