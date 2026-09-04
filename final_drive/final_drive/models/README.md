# Traffic housing ONNX

`traffic_housing_v2_best.onnx` is the single-class four-lamp housing detector
used by `final_drive_vision.traffic_housing_detector`.

- SHA-256: `c3c77b0de3ffc891e9f831ca559c2f8d96904204df30f63ac1fedf8a0c81d169`
- Input: float32 RGB `[1, 3, 640, 640]`, scaled by `1/255`, letterbox value 114
- Output: float32 `[1, 5, 8400]` (`cx, cy, width, height, traffic_light score`)
- Export: YOLOv8n, ONNX opset 12, fixed input, no embedded NMS
- PyTorch exporter: legacy TorchScript path (`dynamo=False`)

Do not replace this with the earlier 12,445,867-byte dynamo export. That graph
fails OpenCV DNN while parsing its first convolution.
