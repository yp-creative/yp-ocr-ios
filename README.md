# yp-ocr-ios
  用于获取照片及检查照片质量，并作预处理。

- [Requirements](#requirements)
- [Usage](#Usage)

## Requirements

- iOS 8.0+
- Xcode 7.0+

## Usage

  将OCR_IOS文件直接引入项目中，该SDK依赖OpenCv，具体使用方法如下：
```objective-c

#import "YP_OCR_IOS.h"

```
在需要使用照片采集的控制器上进行Push操作，同时需要设置照片结果处理Block。
```objective-c

[self.navigationController pushViewController:[YPIDCardOCRManager shareManager].rootViewController animated:YES];

[[YPIDCardOCRManager shareManager] SetPhotoHandler:...];

```
