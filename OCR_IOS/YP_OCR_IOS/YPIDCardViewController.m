//
//  YPIDCardViewController.m
//  OCR_IDCardTest
//
//  Created by yp-tc-m-2548 on 16/7/14.
//  Copyright © 2016年 yp-tc-m-2548. All rights reserved.
//

#import "YPIDCardViewController.h"
#import "YPIDCardOCRManager.h"
#import "YPSystemMacro.h"

@implementation YPIDCardViewController

#pragma mark -Setting

- (void)viewWillAppear:(BOOL)animated{
    [super viewWillAppear:animated];
    [[YPIDCardOCRManager shareManager].session startRunning];

    for (UIBarButtonItem *item in self.toolbarItems) {
        [item setEnabled:YES];
    }
}

- (void)viewDidDisappear:(BOOL)animated{
    [super viewDidDisappear:animated];
    [[YPIDCardOCRManager shareManager].session stopRunning];
}

- (void)viewDidLoad{

    [super viewDidLoad];
    [self.view setBackgroundColor:[UIColor whiteColor]];

    [self setupToolBar];
    
    [self setupNavigationBar];
    
    [self.view.layer addSublayer:[YPIDCardOCRManager shareManager].previewLayer];

//    [self.view addSubview:[self IdCardView]];

}

//- (UIView *)IdCardView{
//
//    UIView *idCardView = [[UIView alloc] initWithFrame:CGRectMake(0, 0, 150, 150)];
//
//    [idCardView.layer setBorderWidth:1];
//    [idCardView.layer setBorderColor:[UIColor blueColor].CGColor];
//
//    [idCardView setCenter:CGPointMake(self.view.center.x, self.view.frame.size.height * 5 / 7)];
//
//    return idCardView;
//}

- (void)setupNavigationBar{
    
    UIBarButtonItem *backItem = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemStop target:self action:@selector(backBarItemAction:)];
    
    [self.navigationItem setLeftBarButtonItem:backItem];
    [self.navigationController.navigationBar setShadowImage:[UIImage new]];
    [self.navigationController.navigationBar setBackgroundImage:[UIImage new] forBarMetrics:UIBarMetricsDefault];
}

- (void)backBarItemAction:(UIBarButtonItem *)sender{
    [self.navigationController popViewControllerAnimated:YES];
}

- (void)setupToolBar{
    
    [self.navigationController setToolbarHidden:NO];

    [self.navigationController.toolbar setFrame:CGRectMake(0, 0, CGRectGetHeight(self.view.frame), ToolBarHeight)];

    [self.navigationController.toolbar setShadowImage:[UIImage new] forToolbarPosition:UIBarPositionBottom];

    [self.navigationController.toolbar setBackgroundImage:[UIImage new] forToolbarPosition:UIBarPositionBottom barMetrics:UIBarMetricsDefault];

    UIBarButtonItem *cameraItem = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemCamera target:self action:@selector(takePhoto:)];

    UIBarButtonItem *albrm = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemBookmarks target:self action:@selector(openAlbrm:)];

    //item 的间隔，不会显示出来,会自动计算间隔
    UIBarButtonItem *spaceItem = [[UIBarButtonItem alloc] initWithBarButtonSystemItem:UIBarButtonSystemItemFlexibleSpace target:self action:nil];

    NSArray *items = [NSArray arrayWithObjects:spaceItem, cameraItem, spaceItem,albrm,spaceItem, nil];

    self.toolbarItems = items;
}

- (UIStatusBarStyle)preferredStatusBarStyle{
    return UIStatusBarStyleDefault;
}

- (BOOL)prefersStatusBarHidden{
    return YES;
}

- (void)takePhoto:(UIBarButtonItem *)sender{
    [[YPIDCardOCRManager shareManager] takePhoto];
}

- (void)openAlbrm:(UIBarButtonItem *)sender{
    [[YPIDCardOCRManager shareManager] openAlbum];
}

@end
