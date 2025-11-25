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

namespace ColorIC_Inspector.Services
{
    // Helper để chạy inference ONNX từ BitmapSource (ví dụ imgCameraFeed.Source as BitmapSource).
    // Trả về Verdict "OK" hoặc "NG" cùng danh sách detections.
    public class YoloOnnxHelper : IDisposable
    {
        public InferenceSession? Session { get; private set; }
        public List<string> ClassNames { get; private set; } = new List<string>();

        public int InputWidth { get; }
        public int InputHeight { get; }
        public float ConfidenceThreshold { get; set; } = 0.25f;
        public float NmsIouThreshold { get; set; } = 0.45f;

        public YoloOnnxHelper(string modelPath, string yamlPath, int inputWidth = 640, int inputHeight = 640, float confThreshold = 0.25f, float nmsIou = 0.45f)
        {
            InputWidth = inputWidth;
            InputHeight = inputHeight;
            ConfidenceThreshold = confThreshold;
            NmsIouThreshold = nmsIou;

            if (!string.IsNullOrEmpty(modelPath) && File.Exists(modelPath))
                Session = new InferenceSession(modelPath);

            if (!string.IsNullOrEmpty(yamlPath) && File.Exists(yamlPath))
                ClassNames = ReadNamesFromYaml(yamlPath);
        }

        // High-level: nhận BitmapSource (imgCameraFeed.Source) và trả về verdict + detections
        public Task<InferenceResult> AnalyzeAsync(BitmapSource bmpSource, CancellationToken cancellationToken = default)
        {
            if (bmpSource == null || Session == null)
            {
                return Task.FromResult(new InferenceResult { Verdict = "OK", Detections = new List<Detection>() });
            }

            // chạy nặng ở thread pool để không block UI
            return Task.Run(() => AnalyzeInternal(bmpSource, cancellationToken), cancellationToken);
        }

        private InferenceResult AnalyzeInternal(BitmapSource bmpSource, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();

            using var bmp = BitmapFromBitmapSource(bmpSource);
            var input = PreprocessBitmap(bmp, InputWidth, InputHeight);

            try
            {
                var inputName = Session!.InputMetadata.Keys.First();
                using var container = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, input) };
                using var results = Session.Run(container);
                var outputs = results.ToList();

                var (boxes, logits) = ExtractOutputs(outputs);
                if (boxes == null || logits == null)
                {
                    return new InferenceResult
                    {
                        Verdict = "OK",
                        Detections = new List<Detection>(),
                        Error = "Unexpected model outputs."
                    };
                }

                var dets = PostprocessOnnxOutputRFDETR(boxes, logits, bmp.Width, bmp.Height, ConfidenceThreshold);
                dets = NonMaxSuppression(dets, NmsIouThreshold);
                string verdict = dets.Any(d => d.Score >= ConfidenceThreshold) ? "NG" : "OK";

                return new InferenceResult { Verdict = verdict, Detections = dets };
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex)
            {
                return new InferenceResult { Verdict = "OK", Detections = new List<Detection>(), Error = ex.Message };
            }
        }

        private (Tensor<float>? boxes, Tensor<float>? logits) ExtractOutputs(IReadOnlyList<NamedOnnxValue> outputs)
        {
            if (outputs.Count < 2) return (null, null);

            Tensor<float>? boxes = null;
            Tensor<float>? logits = null;

            var b = outputs.FirstOrDefault(o => o.Name.ToLower().Contains("box") || o.Name.ToLower().Contains("bbox") || o.Name.ToLower().Contains("boxes"));
            var l = outputs.FirstOrDefault(o => o.Name.ToLower().Contains("logit") || o.Name.ToLower().Contains("class") || o.Name.ToLower().Contains("scores"));
            if (b != null && l != null)
            {
                boxes = b.AsTensor<float>();
                logits = l.AsTensor<float>();
            }
            else
            {
                boxes = outputs[0].AsTensor<float>();
                logits = outputs[1].AsTensor<float>();
            }

            return (boxes, logits);
        }

        // Convert BitmapSource to System.Drawing.Bitmap (BMP stream)
        private Bitmap BitmapFromBitmapSource(BitmapSource source)
        {
            using var ms = new MemoryStream();
            var encoder = new BmpBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(source));
            encoder.Save(ms);
            ms.Seek(0, SeekOrigin.Begin);
            var tmp = new Bitmap(ms);
            // clone to decouple from stream lifetime
            return new Bitmap(tmp);
        }

        // Preprocess (kept similar to original): resize with letterbox, bilinear sampling, output tensor [1,3,H,W]
        private unsafe DenseTensor<float> PreprocessBitmap(Bitmap bitmap, int targetWidth, int targetHeight)
        {
            Bitmap srcBmp = bitmap;
            if (bitmap.PixelFormat != PixelFormat.Format24bppRgb)
            {
                srcBmp = new Bitmap(bitmap.Width, bitmap.Height, PixelFormat.Format24bppRgb);
                using (var g = Graphics.FromImage(srcBmp))
                    g.DrawImage(bitmap, 0, 0, bitmap.Width, bitmap.Height);
            }

            float ratio = Math.Min((float)targetWidth / srcBmp.Width, (float)targetHeight / srcBmp.Height);
            int newWidth = (int)(srcBmp.Width * ratio);
            int newHeight = (int)(srcBmp.Height * ratio);
            int padX = (targetWidth - newWidth) / 2;
            int padY = (targetHeight - newHeight) / 2;

            var input = new DenseTensor<float>(new[] { 1, 3, targetHeight, targetWidth });

            var rect = new System.Drawing.Rectangle(0, 0, srcBmp.Width, srcBmp.Height);
            var bmpData = srcBmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

            try
            {
                byte* srcBase = (byte*)bmpData.Scan0;
                int srcStride = bmpData.Stride;

                for (int y = 0; y < targetHeight; y++)
                {
                    for (int x = 0; x < targetWidth; x++)
                    {
                        float fx = (x - padX + 0.5f) / ratio - 0.5f;
                        float fy = (y - padY + 0.5f) / ratio - 0.5f;

                        float r = 0, g = 0, b = 0;
                        if (fx >= 0 && fx < srcBmp.Width - 1 && fy >= 0 && fy < srcBmp.Height - 1)
                        {
                            int x0 = (int)Math.Floor(fx);
                            int y0 = (int)Math.Floor(fy);
                            int x1 = x0 + 1;
                            int y1 = y0 + 1;
                            float dx = fx - x0;
                            float dy = fy - y0;

                            byte* p00 = srcBase + y0 * srcStride + x0 * 3;
                            byte* p10 = srcBase + y0 * srcStride + x1 * 3;
                            byte* p01 = srcBase + y1 * srcStride + x0 * 3;
                            byte* p11 = srcBase + y1 * srcStride + x1 * 3;

                            b = (1 - dx) * (1 - dy) * p00[0] + dx * (1 - dy) * p10[0] + (1 - dx) * dy * p01[0] + dx * dy * p11[0];
                            g = (1 - dx) * (1 - dy) * p00[1] + dx * (1 - dy) * p10[1] + (1 - dx) * dy * p01[1] + dx * dy * p11[1];
                            r = (1 - dx) * (1 - dy) * p00[2] + dx * (1 - dy) * p10[2] + (1 - dx) * dy * p01[2] + dx * dy * p11[2];

                            b /= 255.0f; g /= 255.0f; r /= 255.0f;
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

        // Postprocess assuming boxes [1,N,4] (cx,cy,w,h normalized) and logits [1,N,C]
        private List<Detection> PostprocessOnnxOutputRFDETR(Tensor<float> boxes, Tensor<float> logits, int origWidth, int origHeight, float confThreshold)
        {
            var dets = new List<Detection>();
            if (boxes == null || logits == null) return dets;

            int numQueries = boxes.Dimensions.Length >= 2 ? boxes.Dimensions[1] : 0;
            int numClasses = logits.Dimensions.Length >= 3 ? logits.Dimensions[2] : logits.Dimensions.Last();

            for (int i = 0; i < numQueries; i++)
            {
                float cx = boxes[0, i, 0];
                float cy = boxes[0, i, 1];
                float w = boxes[0, i, 2];
                float h = boxes[0, i, 3];

                float maxScore = float.MinValue;
                int classId = -1;
                for (int c = 0; c < numClasses; c++)
                {
                    float score = logits[0, i, c];
                    if (score > maxScore)
                    {
                        maxScore = score;
                        classId = c;
                    }
                }

                float confidence = (float)(1.0 / (1.0 + Math.Exp(-maxScore)));
                if (confidence < confThreshold) continue;

                float x1 = (cx - w / 2f) * origWidth;
                float y1 = (cy - h / 2f) * origHeight;
                float x2 = (cx + w / 2f) * origWidth;
                float y2 = (cy + h / 2f) * origHeight;

                dets.Add(new Detection
                {
                    X1 = x1,
                    Y1 = y1,
                    X2 = x2,
                    Y2 = y2,
                    Score = confidence,
                    ClassId = classId,
                    ClassName = (classId >= 0 && classId < ClassNames.Count) ? ClassNames[classId] : null
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
            Session?.Dispose();
            Session = null;
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