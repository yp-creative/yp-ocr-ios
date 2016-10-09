//
//  YPOpencvOperate.h
//  OCR_IDCardTest
//
//  Created by yp-tc-m-2548 on 16/8/2.
//  Copyright © 2016年 yp-tc-m-2548. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

typedef NS_ENUM(NSUInteger, YPBrightnessType) {
    YPBrightnessTypeSuitable,
    YPBrightnessTypehigh,
    YPBrightnessTypelow,
};

@interface YPOpencvOperate : NSObject

+ (UIImage *)thresholdWithImage:(UIImage *)source;

+ (YPBrightnessType)isBrightnessWithImage:(UIImage *)image;

+ (BOOL)isBlurryWithImage:(UIImage *)srcImage thresh:(double)thresh;

@end
