//
//  YPIDCardOCRManager.h
//  OCR_IDCardTest
//
//  Created by yp-tc-m-2548 on 16/7/13.
//  Copyright © 2016年 yp-tc-m-2548. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <UIKit/UIKit.h>

typedef void (^YPPhotoHandler)(UIImage *photo);

@class YPIDCardViewController;

@interface YPIDCardOCRManager : NSObject

@property (nonatomic, strong, readonly) AVCaptureSession *session;

@property (nonatomic, strong, readonly) AVCaptureStillImageOutput *imageOut;

@property (nonatomic, strong, readonly) AVCaptureDeviceInput *input;

@property (nonatomic, strong, readonly) AVCaptureVideoPreviewLayer *previewLayer;

@property (nonatomic, strong) YPIDCardViewController *rootViewController;

@property (nonatomic, copy) YPPhotoHandler photoHandler;

@property (nonatomic, strong, readonly) UIImage *photo;

+ (instancetype)shareManager;

- (void)makePhotoHandler:(YPPhotoHandler)handler;

- (void)takePhoto;

- (void)openAlbum;

@end
