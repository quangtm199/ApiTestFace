from PIL import Image
import numpy
import io
import torch
import cv2
import numpy as np
from MTCNN import *
from torchvision import transforms
from models.CDCNs import Conv2d_cd
from models.Load_OULUNPU_valtest import Normaliztion_valtest_image, ToTensor_valtest_img
from models.CDCNs import CDCNpp
def getmodel():
    pnet = PNet1(test=True)
    rnet = RNet1(test=True)
    onet = ONet1(test=True)
    ctx = mx.cpu()
    pnet.load_parameters('./save_model/pnet1_150000', ctx=ctx)
    pnet.hybridize()
    rnet.load_parameters('./save_model/rnet1_300000', ctx=ctx)
    rnet.hybridize()
    onet.load_parameters('./save_model/onet_80000', ctx=ctx)
    onet.hybridize()
    mtcnn = MTCNN(detectors=[pnet, rnet, onet], min_face_size=24, scalor=0.709, threshold=[0.6, 0.7, 0.7], ctx=ctx)
    if torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    model = CDCNpp(basic_conv=Conv2d_cd, theta=0.7)
    model.to(device)
    model.load_state_dict(torch.load('./models/model.pkl',
                          map_location=torch.device(device)))
    model.eval()
    return model, mtcnn


def get_predict(model, mtcnn, binary_image, max_size=512):
    if torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    input_image = Image.open(io.BytesIO(binary_image)).convert('RGB')
    input_image = numpy.array(input_image)
    # Convert RGB to BGR
    input_image = input_image[:, :, ::-1].copy()
    (h, w, c) = input_image.shape
    image=input_image
    print(input_image.shape)
    bboxes = mtcnn.detect(image)
    if bboxes is not None:
        count = 0
        for i in bboxes:
            lx = int(i[0])
            ly = int(i[1])
            rx = lx + int(i[2])
            ry = ly + int(i[3])
            face = input_image[ly:ry, lx:rx]
            if face.shape[0] < 256 or face.shape[1] < 256:
                center = [int(0.5 * (lx + rx)), int(0.5 * (ly + ry))]
                lx = max(0, center[0] - 128)
                ly = max(0, center[1] - 128)
                rx = min(w, center[0] + 128)
                ry = min(h, center[1] + 128)
                face = input_image[ly:ry, lx:rx]
         
            transform = transforms.Compose(
                [Normaliztion_valtest_image(), ToTensor_valtest_img()])
            img = transform(face)
            img = img.unsqueeze_(0)
            img = img.to(device)
            map_x, embedding, x_Block1, x_Block2, x_Block3, x_input = model(
                img)
            score = torch.sum(map_x)/1024
            if(score > 0.07):
                label = "real"
                spoofing = 0
            else:
                label = "spoofing"
                spoofing = 1
            if (spoofing == 1):
                # Neu la fake thi ve mau do
                cv2.putText(input_image, str(label), (lx, ly - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(input_image, (lx, ly), (rx, ry),
                              (0, 0, 255), 2)
            else:
                # Neu real thi ve mau xanh
                cv2.putText(input_image, str(label), (lx, ly - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(input_image, (lx, ly), (rx, ry),
                              (0, 255, 0), 2)
    # cv2.imwrite("input_image.jpg",input_image)
    input_image = Image.fromarray(np.uint8(input_image)).convert('RGB')
    return input_image
