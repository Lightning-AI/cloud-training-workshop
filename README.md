# cloud-training-workshop


## For training:
    copy the data:
    `aws s3 cp --recursive s3://imagenet-tiny  data`

    remove 24-bytes images:

    `rm data/train/American_chameleon/n01688243_313.JPEG ` 
    `rm data/train/African_elephant/n02504458_3091.JPEG  `