using ColorIC_Inspector;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using YamlDotNet.RepresentationModel;
using static OpenCvSharp.FileStorage;

namespace ColorIC_Inspector.components
{
    // Helper để chạy inference ONNX từ BitmapSource (ví dụ imgCameraFeed.Source as BitmapSource).
    // Trả về Verdict "OK" hoặc "NG" cùng danh sách detections.
    public class YoloOnnxHelper : IDisposable
    {
        public InferenceSession? Session { get; private set; }
        public List<string> ClassNames { get; private set; } = new List<string>();
        private readonly System.Threading.ReaderWriterLockSlim _sessionLock = new System.Threading.ReaderWriterLockSlim();
        private bool _disposed;

        public int InputWidth { get; private set; }
        public int InputHeight { get; private set; }
        public float ConfidenceThreshold { get; set; } = 0.25f;
        public float NmsIouThreshold { get; set; } = 0.45f;


        // Changed: constructor reads model input shape (if available) and sets InputWidth/InputHeight
        public YoloOnnxHelper(string modelPath, string yamlPath, int inputWidth = 640, int inputHeight = 640, float confThreshold = 0.25f, float nmsIou = 0.45f)
        {
            InputWidth = inputWidth;
            InputHeight = inputHeight;
            ConfidenceThreshold = confThreshold;
            NmsIouThreshold = nmsIou;

            if (!string.IsNullOrEmpty(modelPath) && File.Exists(modelPath))
            {
                Session = new InferenceSession(modelPath);

                // Try to read model input spatial dims (common layout: [N,C,H,W])
                TryPopulateInputShapeFromModel();
            }

            if (!string.IsNullOrEmpty(yamlPath) && File.Exists(yamlPath))
                ClassNames = ReadNamesFromYaml(yamlPath);
        }
        private void TryPopulateInputShapeFromModel()
        {
            var firstInput = Session!.InputMetadata.FirstOrDefault();
            if (string.IsNullOrEmpty(firstInput.Key)) return;

            var dims = firstInput.Value.Dimensions;
            if (dims == null || dims.Length < 4) return;

            int h = Math.Max(dims[dims.Length - 2], 1);
            int w = Math.Max(dims[dims.Length - 1], 1);

            // Some models expose dynamic dims (-1). Only override when positive.
            if (h > 0 && w > 0)
            {
                InputWidth = w;
                InputHeight = h;
            }
        }

        // High-level: nhận BitmapSource (imgCameraFeed.Source) và trả về verdict + detections
        public Task<InferenceResult> AnalyzeAsync(
            BitmapSource bmpSource,
            CancellationToken cancellationToken = default)
        {
            if (bmpSource == null)
            {
                return Task.FromResult(new InferenceResult
                {
                    Verdict = "OK",
                    Detections = new List<Detection>()
                });
            }

            // Nếu object đã dispose, không cho chạy nữa
            if (_disposed)
            {
                return Task.FromResult(new InferenceResult
                {
                    Verdict = "OK",
                    Detections = new List<Detection>(),
                    Error = "YoloOnnxHelper has been disposed."
                });
            }

            // RẤT QUAN TRỌNG: Clone + Freeze để dùng được trên background thread
            var clone = bmpSource.Clone();
            clone.Freeze();

            // chạy nặng ở thread pool để không block UI
            return Task.Run(() => AnalyzeInternal(clone, cancellationToken), cancellationToken);
        }

        // --- NEW AnalyzeInternal snippet (adapted to YOLOv12 single-output format) ---
        private InferenceResult AnalyzeInternal(
    BitmapSource bmpSource,
    CancellationToken cancellationToken)
        {
            _sessionLock.EnterReadLock();
            try
            {
                // Nếu đã dispose hoặc Session null thì không chạy gì nữa
                if (_disposed || Session == null)
                {
                    return new InferenceResult
                    {
                        Verdict = "OK",
                        Detections = new List<Detection>(),
                        Error = "Session is null or disposed."
                    };
                }

                cancellationToken.ThrowIfCancellationRequested();

                // Convert BitmapSource -> Bitmap
                using var bmp = BitmapFromBitmapSource(bmpSource);
                cancellationToken.ThrowIfCancellationRequested();

                var input = PreprocessBitmap(
                    bmp, InputWidth, InputHeight,
                    out float ratio, out float padX, out float padY);
                cancellationToken.ThrowIfCancellationRequested();

                var inputName = Session.InputMetadata.Keys.First();

                // đảm bảo NamedOnnxValue được dispose đúng cách
                var named = NamedOnnxValue.CreateFromTensor(inputName, input);
                // chạy inference
                using var results = Session.Run(new[] { named });

                var output = results.First().AsTensor<float>();

                var dets = PostprocessOnnxOutput(
                    output, bmp.Width, bmp.Height,
                    InputWidth, InputHeight, ratio, padX, padY, ConfidenceThreshold);

                dets = NonMaxSuppression(dets, NmsIouThreshold);

                string verdict = dets.Any(d => d.Score >= ConfidenceThreshold) ? "NG" : "OK";

                return new InferenceResult
                {
                    Verdict = verdict,
                    Detections = dets
                };
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex)
            {
                return new InferenceResult
                {
                    Verdict = "OK",
                    Detections = new List<Detection>(),
                    Error = ex.Message
                };
            }
            finally
            {
                _sessionLock.ExitReadLock();
            }
        }

        // --- Updated method: declare locals before assignment ---
        // Convert BitmapSource to System.Drawing.Bitmap (BMP stream)
        private Bitmap BitmapFromBitmapSource(BitmapSource source)
        {
            using var ms = new MemoryStream();
            var encoder = new BmpBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(source));
            encoder.Save(ms);
            ms.Seek(0, SeekOrigin.Begin);

            using var tmp = new Bitmap(ms);
            return new Bitmap(tmp); // clone để tách khỏi stream
        }


        // Preprocess (kept similar to original): resize with letterbox, bilinear sampling, output tensor [1,3,H,W]
        // Changed: replace unsafe PreprocessBitmap with safe Marshal.Copy-based implementation to avoid pointer issues
        private DenseTensor<float> PreprocessBitmap(Bitmap bitmap, int targetWidth, int targetHeight, out float ratio, out float padX, out float padY)
        {
            Bitmap srcBmp = bitmap;
            if (bitmap.PixelFormat != PixelFormat.Format24bppRgb)
            {
                srcBmp = new Bitmap(bitmap.Width, bitmap.Height, PixelFormat.Format24bppRgb);
                using (var g = Graphics.FromImage(srcBmp))
                    g.DrawImage(bitmap, 0, 0, bitmap.Width, bitmap.Height);
            }

            ratio = Math.Min((float)targetWidth / srcBmp.Width, (float)targetHeight / srcBmp.Height);
            int newWidth = (int)(srcBmp.Width * ratio);
            int newHeight = (int)(srcBmp.Height * ratio);
            padX = (targetWidth - newWidth) / 2;
            padY = (targetHeight - newHeight) / 2;

            var input = new DenseTensor<float>(new[] { 1, 3, targetHeight, targetWidth });

            var rect = new System.Drawing.Rectangle(0, 0, srcBmp.Width, srcBmp.Height);
            var bmpData = srcBmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

            try
            {
                int srcStride = Math.Abs(bmpData.Stride);
                int srcLen = srcStride * srcBmp.Height;
                byte[] src = new byte[srcLen];
                System.Runtime.InteropServices.Marshal.Copy(bmpData.Scan0, src, 0, srcLen);

                for (int y = 0; y < targetHeight; y++)
                {
                    for (int x = 0; x < targetWidth; x++)
                    {
                        float fx = (x - padX + 0.5f) / ratio - 0.5f;
                        float fy = (y - padY + 0.5f) / ratio - 0.5f;

                        float r = 0f, g = 0f, b = 0f;
                        if (fx >= 0 && fx < srcBmp.Width - 1 && fy >= 0 && fy < srcBmp.Height - 1)
                        {
                            int x0 = (int)Math.Floor(fx);
                            int y0 = (int)Math.Floor(fy);
                            int x1 = x0 + 1;
                            int y1 = y0 + 1;
                            float dx = fx - x0;
                            float dy = fy - y0;

                            int idx00 = y0 * srcStride + x0 * 3;
                            int idx10 = y0 * srcStride + x1 * 3;
                            int idx01 = y1 * srcStride + x0 * 3;
                            int idx11 = y1 * srcStride + x1 * 3;

                            byte p00b = src[idx00 + 0];
                            byte p00g = src[idx00 + 1];
                            byte p00r = src[idx00 + 2];

                            byte p10b = src[idx10 + 0];
                            byte p10g = src[idx10 + 1];
                            byte p10r = src[idx10 + 2];

                            byte p01b = src[idx01 + 0];
                            byte p01g = src[idx01 + 1];
                            byte p01r = src[idx01 + 2];

                            byte p11b = src[idx11 + 0];
                            byte p11g = src[idx11 + 1];
                            byte p11r = src[idx11 + 2];

                            b = (1 - dx) * (1 - dy) * p00b + dx * (1 - dy) * p10b + (1 - dx) * dy * p01b + dx * dy * p11b;
                            g = (1 - dx) * (1 - dy) * p00g + dx * (1 - dy) * p10g + (1 - dx) * dy * p01g + dx * dy * p11g;
                            r = (1 - dx) * (1 - dy) * p00r + dx * (1 - dy) * p10r + (1 - dx) * dy * p01r + dx * dy * p11r;

                            b /= 255f; g /= 255f; r /= 255f;
                        }

                        input[0, 0, y, x] = r;
                        input[0, 1, y, x] = g;
                        input[0, 2, y, x] = b;
                    }
                }
            }
            finally
            {
                srcBmp.UnlockBits(bmpData);
                if (!object.ReferenceEquals(srcBmp, bitmap))
                    srcBmp.Dispose();
            }

            return input;
        }
        private List<Detection> PostprocessOnnxOutput(Tensor<float> output, int origWidth, int origHeight, int inputWidth, int inputHeight, float ratio, float padX, float padY, float confThreshold)
        {
            var dets = new List<Detection>();
            if (output == null) return dets;

            int[] dims = output.Dimensions.ToArray();
            // dims = [1, features, num_boxes] or [1, num_boxes, features] depending on model; handle common case [1, features, num_boxes]
            if (dims.Length != 3 || dims[0] != 1) return dets;

            int features = dims[1];
            int numBoxes = dims[2];

            // If model uses [1, num_boxes, features], detect and transpose logically
            bool transposed = false;
            if (features < 5 && dims[1] != 5 && dims[2] >= 5)
            {
                // likely [1, num_boxes, features]
                transposed = true;
                features = dims[2];
                numBoxes = dims[1];
            }

            // Compute ratio/pad used in preprocessing (letterbox)
            if (ratio <= 0f)
            {
                ratio = Math.Min((float)inputWidth / origWidth, (float)inputHeight / origHeight);
                padX = (inputWidth - origWidth * ratio) / 2f;
                padY = (inputHeight - origHeight * ratio) / 2f;
            }

            for (int i = 0; i < numBoxes; i++)
            {
                float cx, cy, w, h, conf;
                // read depending on layout
                if (!transposed)
                {
                    cx = output[0, 0, i];
                    cy = output[0, 1, i];
                    w = output[0, 2, i];
                    h = output[0, 3, i];
                    conf = output[0, 4, i];
                }
                else
                {
                    cx = output[0, i, 0];
                    cy = output[0, i, 1];
                    w = output[0, i, 2];
                    h = output[0, i, 3];
                    conf = output[0, i, 4];
                }

                if (conf < confThreshold) continue;

                // If classes exist after conf, find class id with highest score
                int classId = 0;
                string? className = null;
                float bestClassScore = 0f;
                int classFeatureStart = 5;

                if (!transposed)
                {
                    if (features > classFeatureStart)
                    {
                        int classCount = features - classFeatureStart;
                        for (int c = 0; c < classCount; c++)
                        {
                            float score = output[0, classFeatureStart + c, i];
                            if (score > bestClassScore) { bestClassScore = score; classId = c; }
                        }
                    }
                }
                else
                {
                    if (features > classFeatureStart)
                    {
                        int classCount = features - classFeatureStart;
                        for (int c = 0; c < classCount; c++)
                        {
                            float score = output[0, i, classFeatureStart + c];
                            if (score > bestClassScore) { bestClassScore = score; classId = c; }
                        }
                    }
                }

                if (bestClassScore > 0 && ClassNames != null && classId >= 0 && classId < ClassNames.Count)
                    className = ClassNames[classId];

                // Convert from model coordinates (assumed in inputSize space, center-based) back to original image
                float x1 = (cx - w / 2f - padX) / ratio;
                float y1 = (cy - h / 2f - padY) / ratio;
                float x2 = (cx + w / 2f - padX) / ratio;
                float y2 = (cy + h / 2f - padY) / ratio;

                dets.Add(new Detection
                {
                    X1 = x1,
                    Y1 = y1,
                    X2 = x2,
                    Y2 = y2,
                    Score = conf,
                    ClassId = classId,
                    ClassName = className
                });
            }

            return dets;
        }

        // NMS grouped by class
        public List<Detection> NonMaxSuppression(List<Detection> dets, float iouThreshold)
        {
            var keep = new List<Detection>();
            var groups = dets.GroupBy(d => d.ClassId);

            foreach (var group in groups)
            {
                var list = group.OrderByDescending(d => d.Score).ToList();
                var suppressed = new bool[list.Count];

                for (int i = 0; i < list.Count; i++)
                {
                    if (suppressed[i]) continue;
                    var chosen = list[i];
                    keep.Add(chosen);

                    for (int j = i + 1; j < list.Count; j++)
                    {
                        if (suppressed[j]) continue;
                        float iou = ComputeIoU(chosen, list[j]);
                        if (iou > iouThreshold)
                            suppressed[j] = true;
                    }
                }
            }
            return keep;
        }

        public float ComputeIoU(Detection a, Detection b)
        {
            float x1 = Math.Max(a.X1, b.X1);
            float y1 = Math.Max(a.Y1, b.Y1);
            float x2 = Math.Min(a.X2, b.X2);
            float y2 = Math.Min(a.Y2, b.Y2);

            float w = Math.Max(0, x2 - x1);
            float h = Math.Max(0, y2 - y1);
            float inter = w * h;

            float areaA = Math.Max(0, (a.X2 - a.X1)) * Math.Max(0, (a.Y2 - a.Y1));
            float areaB = Math.Max(0, (b.X2 - b.X1)) * Math.Max(0, (b.Y2 - b.Y1));

            return inter / (areaA + areaB - inter + 1e-6f);
        }

        private List<string> ReadNamesFromYaml(string yamlFile)
        {
            var names = new List<string>();
            using var reader = new StreamReader(yamlFile);
            var yaml = new YamlStream();
            yaml.Load(reader);
            var mapping = (YamlMappingNode)yaml.Documents[0].RootNode;

            if (mapping.Children.ContainsKey(new YamlScalarNode("names")))
            {
                var namesNode = mapping.Children[new YamlScalarNode("names")];
                if (namesNode is YamlSequenceNode seq)
                {
                    foreach (YamlNode n in seq.Children)
                    {
                        string val = ((YamlScalarNode)n).Value?.Trim('\'', '"') ?? string.Empty;
                        names.Add(val);
                    }
                }
            }
            return names;
        }

        public void Dispose()
        {
            _sessionLock.EnterWriteLock();
            try
            {
                if (_disposed)
                    return;

                Session?.Dispose();
                Session = null;
                _disposed = true;
            }
            finally
            {
                _sessionLock.ExitWriteLock();
                // KHÔNG bắt buộc dispose _sessionLock ở đây,
                // trừ khi bạn chắc chắn không ai dùng helper nữa.
                // _sessionLock.Dispose();
            }
        }

    }

    // Simple result DTO
    public class InferenceResult
    {
        public string Verdict { get; set; } = "OK"; // "OK" or "NG"
        public List<Detection> Detections { get; set; } = new List<Detection>();
        public string? Error { get; set; }
    }
}