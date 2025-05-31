import argparse
from ultralytics import YOLO

def main(opt):
    # Load model
    model = YOLO(opt.model)

    # Train model
    model.train(
        data=opt.data,
        epochs=opt.epochs,
        imgsz=opt.imgsz,
        batch=opt.batch,
        name=opt.name,
        project=opt.project,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolo12n.pt', help='Pretrained model path (e.g. yolov8n.pt, yolov12n.pt)')
    parser.add_argument('--data', type=str, default='traffic-vio-detect/data.yaml', help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--name', type=str, default='yolov12-custom', help='Experiment name')
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory')

    opt = parser.parse_args()
    main(opt)
