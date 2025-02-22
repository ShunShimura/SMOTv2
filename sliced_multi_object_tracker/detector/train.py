from sliced_multi_object_tracker.detector.ultralytics import YOLOv10
import os, sys
import argparse

'''
CAUTION: This script must be runned on SMOT directory.

library: (conda 24.9.2)
    conda install pytorch torchvision torchaudio cudatoolkit=12.1
    opencv: CAUTION; opencv may change pytorch version from one for GPU to one for CPU
    psutil
    matplotlib
    tqdm 
    pandas
    huggingface_hub (Please pay attension to "_" ! Not "-" !)
    -> please install simultanously !
    
'''

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--weights", type=str, default='sliced_multi_object_tracker/detector/ultralytics/pretrained/yolov10n.pt')
    return parser.parse_args()

if __name__=='__main__':
    opt = parse_opt()
    opt.device = [int(val) for val in opt.device.split(",")]
    
    model = YOLOv10(opt.weights)
    
    model.train(data=opt.data,
                epochs=opt.epochs,
                batch=opt.batch_size,
                imgsz=opt.imgsz,
                device=opt.device,
                name=opt.name,
                exist_ok=opt.exist_ok
                )
