//
//  YPIDCardOCRManager.m
//  OCR_IDCardTest
//
//  Created by yp-tc-m-2548 on 16/7/13.
//  Copyright © 2016年 yp-tc-m-2548. All rights reserved.
//

#import "YPIDCardOCRManager.h"
#import "YPIDCardViewController.h"
#import <Photos/Photos.h>
#import "YPSystemMacro.h"
#import "YPOpencvOperate.h"
//#import <JGProgressHUD/JGProgressHUD.h>

#define RectPreviewLayer CGRectMake(0, 0, _rootViewController.view.frame.size.width, _rootViewController.view.frame.size.height)

#define ShadowSpace NavigationBarHeight_Por

@interface YPIDCardOCRManager () <AVCaptureMetadataOutputObjectsDelegate,AVCaptureVideoDataOutputSampleBufferDelegate,UIImagePickerControllerDelegate,UINavigationControllerDelegate>

@property (nonatomic, strong) AVCaptureSession *session;

@property (nonatomic, strong) AVCaptureStillImageOutput *imageOut;

@property (nonatomic, strong) AVCaptureMetadataOutput *metadaOut;

@property (nonatomic, strong) AVCaptureVideoDataOutput *videoOut;

@property (nonatomic, strong) AVCaptureDeviceInput *input;

@property (nonatomic, strong) AVCaptureVideoPreviewLayer *previewLayer;

@property (nonatomic, strong) UIAlertController *alertController;

@property (nonatomic, strong) AVCaptureDevice *device;

@property (nonatomic, strong) CAShapeLayer *shapeLayer;

@property (nonatomic, strong) UIImage *photo;

@property (nonatomic, assign) NSNumber *outputSetting;

@property (nonatomic, strong) UIToolbar *toolBar;

@property (nonatomic, assign) unsigned char* buffer;

@property (nonatomic, assign) BOOL confirmResult;

@end

@implementation YPIDCardOCRManager

#pragma mark - Init Instance

- (void)makePhotoHandler:(YPPhotoHandler)handler{
    self.photoHandler = handler;
}

+ (instancetype)shareManager{

    static YPIDCardOCRManager *manager ;

    static dispatch_once_t onceToken;

    dispatch_once(&onceToken, ^{
        
        manager.confirmResult = NO;
        
        manager = [[YPIDCardOCRManager alloc] init];
        
        [manager.device lockForConfiguration:nil];
        
        if (manager.device.hasTorch)
            [manager.device setTorchMode:AVCaptureTorchModeOff];
        
        [manager.device setFocusMode:AVCaptureFocusModeContinuousAutoFocus];
        [manager.device unlockForConfiguration];
        
        if ([manager.session canAddInput:manager.input]) {
            [manager.session addInput:manager.input];
        }
        if ([manager.session canAddOutput:manager.imageOut]) {
            [manager.session addOutput:manager.imageOut];
        }
        //        if ([manager.session canAddOutput:manager.videoOut]) {
        //            [manager.session addOutput:manager.videoOut];
        //        }
    });

    return manager;
}

#pragma mark - Init Property

- (AVCaptureSession *)session{
    
    if (!_session) {
        _session = [[AVCaptureSession alloc] init];
        [_session setSessionPreset:AVCaptureSessionPreset1280x720];
    }
    
    return _session;
}

- (AVCaptureDevice *)device{
    
    if (!_device) {
        _device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
//        [_device lockForConfiguration:nil];
//        [_device setActiveVideoMinFrameDuration:CMTimeMake(1, 15)];
//        [_device unlockForConfiguration];
    }
    
    return _device;
}

- (AVCaptureDeviceInput *)input{
    
    if (!_input) {
        NSError *error;
        _input = [[AVCaptureDeviceInput alloc] initWithDevice:self.device error:&error];
        if (error) {
            NSLog(@"%@",error);
        }
    }
    
    return _input;
}

- (AVCaptureStillImageOutput *)imageOut{
    
    if (!_imageOut) {
        _imageOut = [[AVCaptureStillImageOutput alloc] init];
        [_imageOut setOutputSettings:@{AVVideoCodecKey:AVVideoCodecJPEG}];
    }

    return _imageOut;
}

- (UIAlertController *)alertController{
    if (!_alertController) {
        UIAlertAction *action = [UIAlertAction actionWithTitle:@"确认" style:UIAlertActionStyleCancel handler:^(UIAlertAction * _Nonnull action) {
            [_alertController dismissViewControllerAnimated:YES completion:nil];
            
            for (UIBarButtonItem *item in self.rootViewController.toolbarItems) {
                [item setEnabled:YES];
            }
            
            [self.session startRunning];
            
        }];
        _alertController = [UIAlertController alertControllerWithTitle:@"提示"
                                                               message:@"图片不合格"
                                                        preferredStyle:UIAlertControllerStyleAlert];
        [_alertController addAction:action];
    }
    return _alertController;
}


- (AVCaptureVideoDataOutput *)videoOut{
    
    if (!_videoOut) {
        
        _videoOut = [[AVCaptureVideoDataOutput alloc] init];
        [_videoOut setVideoSettings:@{(NSString *)kCVPixelBufferPixelFormatTypeKey : self.outputSetting}];
        [_videoOut setAlwaysDiscardsLateVideoFrames:YES];
        dispatch_queue_t queue = dispatch_queue_create("ViewQueue", NULL);
        [_videoOut setSampleBufferDelegate:self queue:queue];
        
    }
    return _videoOut;
}

- (AVCaptureMetadataOutput *)metadaOut{
    
    if (!_metadaOut) {
        _metadaOut = [[AVCaptureMetadataOutput alloc] init];
        [_metadaOut setRectOfInterest:RectPreviewLayer];
        dispatch_queue_t queue = dispatch_queue_create("MetadaOut", NULL);
        [_metadaOut setMetadataObjectsDelegate:self queue:queue];
    }

    return _metadaOut;
}

- (NSNumber *)outputSetting{

    if (!_outputSetting) {
        _outputSetting = [NSNumber numberWithInt:kCVPixelFormatType_420YpCbCr8BiPlanarFullRange];
    }

    return _outputSetting;
}

- (AVCaptureVideoPreviewLayer *)previewLayer{
    
    if (!_previewLayer) {

        _previewLayer = [AVCaptureVideoPreviewLayer layerWithSession:self.session];
        [_previewLayer setFrame:RectPreviewLayer];
        [_previewLayer setVideoGravity:AVLayerVideoGravityResizeAspectFill];

        CALayer *maskLayer = [CALayer layer];
        [maskLayer setFrame:RectPreviewLayer];
        [maskLayer setBackgroundColor:[UIColor colorWithWhite:0 alpha:0.8].CGColor];
        [maskLayer setMask:self.shapeLayer];
        [_previewLayer addSublayer:maskLayer];

    }

    return _previewLayer;
}

- (YPIDCardViewController *)rootViewController{

    if (!_rootViewController) {
        _rootViewController = [[YPIDCardViewController alloc] init];
        _rootViewController.navigationItem.rightBarButtonItem = [[UIBarButtonItem alloc] initWithImage:[UIImage imageNamed:@"flash_off.png"] style:UIBarButtonItemStylePlain target:self action:@selector(flashButtonAction:)];
    }
    return _rootViewController;
}

- (CAShapeLayer *)shapeLayer{

    if (!_shapeLayer) {

        _shapeLayer = [CAShapeLayer layer];

        UIBezierPath *shapePath = [UIBezierPath bezierPathWithRect:RectPreviewLayer];

        [shapePath appendPath:[[UIBezierPath bezierPathWithRect:
                                CGRectMake(0, ShadowSpace,
                                           CGRectGetWidth(RectPreviewLayer), CGRectGetHeight(RectPreviewLayer))] bezierPathByReversingPath]];
        _shapeLayer.path = shapePath.CGPath;
    }

    return _shapeLayer;
}

#pragma mark Setting

- (AVCaptureVideoOrientation)videoOrientationOfDevice{
    UIDeviceOrientation ori = [UIDevice currentDevice].orientation;
    return (AVCaptureVideoOrientation)ori;
}

- (void)takePhoto{
    
    for (UIBarButtonItem *item in self.rootViewController.toolbarItems) {
        [item setEnabled:NO];
    }

    AVCaptureConnection *imageConnect = [self.imageOut connectionWithMediaType:AVMediaTypeVideo];

    [imageConnect setVideoOrientation:[self videoOrientationOfDevice]];

    __weak typeof(self) weakSelf = self;

    [self.imageOut captureStillImageAsynchronouslyFromConnection:imageConnect
                                               completionHandler:^(CMSampleBufferRef imageDataSampleBuffer, NSError *error)
    {
        if (imageDataSampleBuffer) {

            NSData *imageData = [AVCaptureStillImageOutput jpegStillImageNSDataRepresentation:imageDataSampleBuffer];

            UIImage *image = [UIImage imageWithData:imageData];

            PHAuthorizationStatus status = [PHPhotoLibrary authorizationStatus];

            [weakSelf.session stopRunning];

            if (status == PHAuthorizationStatusDenied || status == PHAuthorizationStatusRestricted)
                return ;

            if ([YPOpencvOperate isBlurryWithImage:image thresh:14] &&
                [YPOpencvOperate isBrightnessWithImage:image]) {

                [weakSelf handlerImage:image];

                for (UIBarButtonItem *item in weakSelf.rootViewController.toolbarItems) {
                    [item setEnabled:YES];
                }

            }else{
                [self.rootViewController presentViewController:self.alertController
                                                      animated:YES
                                                    completion:nil];
            }
        }else{
            NSLog(@"error:%@",error);
        }
    }];
}

- (void)openAlbum{

    UIImagePickerController *albumPicker = [[UIImagePickerController alloc] init];
    albumPicker.sourceType = UIImagePickerControllerSourceTypePhotoLibrary;
    albumPicker.delegate = self;

    [albumPicker setAllowsEditing:YES];
    //设置选择后的图片可被编辑
    albumPicker.allowsEditing = NO;

    [self.rootViewController presentViewController:albumPicker animated:YES completion:^{
        [self.session stopRunning];
    }];

}

- (void)flashButtonAction:(UIButton *)sender{

    [self.device lockForConfiguration:nil];

    if (self.device.torchAvailable) {
        self.device.torchMode = (self.device.torchMode + 1) % 2;
        switch (self.device.torchMode) {
            case AVCaptureTorchModeOff: {
                self.rootViewController.navigationItem.rightBarButtonItem = [[UIBarButtonItem alloc] initWithImage:[UIImage imageNamed:@"flash_off.png"] style:UIBarButtonItemStylePlain target:self action:@selector(flashButtonAction:)];
                break;
            }
            case AVCaptureTorchModeOn: {
                self.rootViewController.navigationItem.rightBarButtonItem = [[UIBarButtonItem alloc] initWithImage:[UIImage imageNamed:@"flash_on.png"] style:UIBarButtonItemStylePlain target:self action:@selector(flashButtonAction:)];
                break;
            }
           default:
                break;
        }
    }else{
        [self.rootViewController presentViewController:
         [UIAlertController alertControllerWithTitle:@"提示"
                                             message:@"无法使用闪光灯"
                                      preferredStyle:UIAlertControllerStyleAlert]
                                              animated:YES
                                            completion:nil];
    }
    [self.device unlockForConfiguration];
}

#pragma mark - Delegate

//- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection{
//    
//    static int count = 0;
//    if (++count % 2) return;
//
//    UIImage *image = [self imageFromSamplePlanerPixelBuffer:sampleBuffer];
//    
//    CGRect faceRect = [YPOpencvOperate faceDetectorRectWithImage:image];
//    
//    if (!CGRectIsNull(faceRect)) {
//        [self.session stopRunning];
//        dispatch_sync(dispatch_get_main_queue(), ^{
//         [self handlerImage:image];
//        });
//    }
//}

//-(UIImage *)imageFromSamplePlanerPixelBuffer:(CMSampleBufferRef) sampleBuffer{
//    // Get a CMSampleBuffer's Core Video image buffer for the media data
//    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
//    // Lock the base address of the pixel buffer
//    CVPixelBufferLockBaseAddress(imageBuffer, 0);
//
//    // Get the number of bytes per row for the plane pixel buffer
//    void *baseAddress = CVPixelBufferGetBaseAddressOfPlane(imageBuffer, 0);
//
//    // Get the number of bytes per row for the plane pixel buffer
//    size_t bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(imageBuffer,0);
//    // Get the pixel buffer width and height
//    size_t width = CVPixelBufferGetWidth(imageBuffer);
//    size_t height = CVPixelBufferGetHeight(imageBuffer);
//
//    // Create a device-dependent gray color space
//    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
//
//    // Create a bitmap graphics context with the sample buffer data
//    CGContextRef context = CGBitmapContextCreate(baseAddress, width, height, 8,
//                                                     bytesPerRow, colorSpace, kCGImageAlphaNone);
//    // Create a Quartz image from the pixel data in the bitmap graphics context
//    CGImageRef quartzImage = CGBitmapContextCreateImage(context);
//    // Unlock the pixel buffer
//    CVPixelBufferUnlockBaseAddress(imageBuffer,0);
//
//    // Free up the context and color space
//    CGContextRelease(context);
//    CGColorSpaceRelease(colorSpace);
//
//    // Create an image object from the Quartz image
//    UIImage *image = [UIImage imageWithCGImage:quartzImage];
//
//    // Release the Quartz image
//    CGImageRelease(quartzImage);
//
//    return (image);
//}

- (void)handlerImage:(UIImage *)image{

    self.photo = image;

    if (self.photoHandler) {
        self.photoHandler(image);
    }

    [self.session startRunning];
}

- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary<NSString *,id> *)info{
    UIImage *image = info[UIImagePickerControllerOriginalImage];
    [picker dismissViewControllerAnimated:YES completion:^{
        if ([YPOpencvOperate isBlurryWithImage:image thresh:14] &&
            [YPOpencvOperate isBrightnessWithImage:image]) {
            [self handlerImage:image];
        }else{
            [picker presentViewController:self.alertController
                                 animated:YES
                               completion:nil];
        }
    }];
}

@end
