# Traffic Violation Detection System using YOLOv12

This project detects traffic violations such as running red lights and stopping past pedestrian crossing lines using YOLOv12. It supports training on custom datasets and provides an inference interface with Gradio for real-time detection.

# Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
````

# Training

Run training with the command:

```bash
python train.py --model yolov12n.pt --data datasets/data.yaml --epochs 50 --batch 16 --imgsz 640 --name yolov12-custom --project runs/train
```

# Running Inference (Main)

Run the main script to load the trained model and launch the Gradio interface:

```bash
python main.py
```

Result in violation_result.json

# Notes

* Make sure your dataset configuration file `datasets/data.yaml` is correctly set up with class names and paths.
* Adjust batch size and epochs according to your hardware.
* For GPU usage, install the compatible PyTorch version from [https://pytorch.org](https://pytorch.org).
* Trained model weights will be saved at `runs/train/yolov12-custom/weights/best.pt`.

