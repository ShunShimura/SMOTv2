
for size in n s m b l x
do
    nohup python sliced_multi_object_tracker/detector/valid.py --source ../Data/cell_dataset/val --conf 0.01 --imgsz 2048 --device 3 --output sliced_multi_object_tracker/detector/valid/cell_size-$size --weights runs/detect/cell_size-$size/weights/best.pt > sliced_multi_object_tracker/detector/logs/cell_size-$size.log
done