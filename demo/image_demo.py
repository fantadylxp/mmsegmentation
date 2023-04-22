# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default="./data/VOCdevkit_aug/VOC2012/JPEGImages/voc_0173.jpg",help='Image file')
    parser.add_argument('--config', default="./work_dirs/deeplabv3_r50-d8_4xb4-20k_voc12aug-512x512/deeplabv3_r50-d8_4xb4-20k_voc12aug-512x512.py",help='Config file')
    parser.add_argument('--checkpoint',default= "/home/bjtds/mmlab/mmsegmentation/work_dirs/deeplabv3_r50-d8_4xb4-20k_voc12aug-512x512/iter_20000.pth",help='Checkpoint file')
    parser.add_argument('--out-file', default="./work_dirs/deeplabv3_r50-d8_4xb4-20k_voc12aug-512x512/test.jpg", help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    result = inference_model(model, args.img)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        title=args.title,
        opacity=args.opacity,
        draw_gt=False,
        show=False if args.out_file is not None else True,
        out_file=args.out_file)


if __name__ == '__main__':
    main()
