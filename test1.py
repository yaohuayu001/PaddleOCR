import paddle

paddle.utils.run_check()


from paddleocr import PaddleOCR
 
ocr = PaddleOCR(use_gpu=True,lang='ch',use_angle_cls=True)  # need to run only once to download and load model into memory
img_path = 'doc/imgs_words/en/yes.png'
result = ocr.ocr(img_path, det=False)
for line in result:
    print(line)
