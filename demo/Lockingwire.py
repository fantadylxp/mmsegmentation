import mmcv
import os
import numpy as np
import matplotlib.pyplot as plt
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, inference_model, show_result_pyplot
from PIL import Image
import cv2 
import imutils
from imutils import contours
class LockingWire():
    def __init__(self,config_file_,checkpoint_file_) -> None:
        config_file =config_file_
        checkpoint_file =checkpoint_file_
        self.model = init_model(config_file, checkpoint_file, device='cuda:0')
    def inference(self,img):
        img=np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h_src, w_src = img.shape[:2]
        result = inference_model(self.model,img)
    def post_process_left(self, img_seg):
        # 背景

        # 螺栓
        mask_bolt = np.zeros_like(img_seg)
        mask_bolt[img_seg == 2] = 255
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # dst_line = cv2.erode(mask_bolt, kernel)  # 腐蚀
        # dst_line = cv2.dilate(dst_line, kernel1)  # 膨胀
        # dst_line = cv2.dilate(dst_line, kernel1)  # 膨胀

        # 铁丝
        # mask_line_ = np.zeros_like(img_seg)
        # mask_line_[seg_ == 1] = 255
        mask_line = np.zeros_like(img_seg)
        mask_line[img_seg == 1] = 255
        # mask_line=cv2.bitwise_or(dst_line ,mask_line)



        # 二值化
        ret_line, thresh_line = cv2.threshold(mask_line, 127, 255, cv2.THRESH_BINARY)
        ret_bolt, thresh_bolt = cv2.threshold(mask_bolt, 127, 255, cv2.THRESH_BINARY)

        # p膨胀操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dst_line = cv2.erode(thresh_line, kernel)  # 腐蚀
        dst_line = cv2.dilate(dst_line, kernel1)  # 膨胀
        dst_line = cv2.dilate(dst_line, kernel1)  # 膨胀
        dst_line = cv2.dilate(dst_line, kernel1)  # 膨胀
        # dst_line = cv2.dilate(thresh_line, kernel)
        cv2.imshow("dst_line",dst_line)
        cv2.waitKey(1000)
        judge_line = 'iron_line_normal'
            # 螺栓连通域
        cnts=cv2.findContours(dst_line,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts=imutils.grab_contours(cnts)
        if len(cnts)>1:
             judge_line = 'iron_line_break'
        return judge_line



if __name__=="__main__":
    test=LockingWire(config_file_ = "/home/bjtds/mmlab/mmsegmentation/work_dirs/deeplabv3_r50-d8_4xb4-20k_voc12aug-512x512/deeplabv3_r50-d8_4xb4-20k_voc12aug-512x512.py",
    checkpoint_file_ = "/home/bjtds/mmlab/mmsegmentation/work_dirs/deeplabv3_r50-d8_4xb4-20k_voc12aug-512x512/iter_20000.pth")
    test_=LockingWire(config_file_ = "/home/bjtds/mmlab/mmsegmentation/work_dirs/deeplabv3_r50-d8_4xb4-20k_voc12aug-512x512_together/deeplabv3_r50-d8_4xb4-20k_voc12aug-512x512.py",
    checkpoint_file_ = "/home/bjtds/mmlab/mmsegmentation/work_dirs/deeplabv3_r50-d8_4xb4-20k_voc12aug-512x512_together/iter_20000.pth")
    for file in os.listdir("/home/bjtds/mmlab/mmsegmentation/demo/test_crop"):
        if file[-4:]==".jpg":
            img=np.array(Image.open(os.path.join("/home/bjtds/mmlab/mmsegmentation/demo/test_crop",file)) )
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            h_src, w_src = img.shape[:2]
            # img = cv2.resize(img, dsize=(512, 512))
            result = inference_model(test.model, img)
            result_=inference_model(test_.model, img)
            seg=np.array(result.pred_sem_seg.numpy().get("data")[0],dtype=np.uint8)
            seg_=np.array(result_.pred_sem_seg.numpy().get("data")[0],dtype=np.uint8)
            
            # seg = cv2.resize(seg, dsize=(w_src, h_src))
            LW_result=test.post_process(seg)
            print(LW_result)
            
            

