## Expected dataset structure for COCO instance/keypoint detection:

```
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

See [detectron2/datasets](https://github.com/facebookresearch/detectron2/tree/e49c555a046a7495db58d327f34058e7dc858275/datasets) for more details.