from fastapi import FastAPI, File,UploadFile
from typing import Optional
import io
import uvicorn
from fakeantispoffing import get_predict, getmodel
from starlette.responses import Response
model ,face_detector = getmodel()
app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}
# @app.post("/files/")
# async def create_file(file: bytes = File(...)):
#     return {"file_size": len(file)}

# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile = File(...)):
#     return {"filename": file.filename}
    
# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}

# @app.post("/create_file/")
# async def image(image: UploadFile = File(...)):
#     print(image.file)
#     # print('../'+os.path.isdir(os.getcwd()+"images"),"*************")
#     try:
#         os.mkdir("images")
#         print(os.getcwd())
#     except Exception as e:
#         print(e) 
#     file_name = os.getcwd()+"/images/"+image.filename.replace(" ", "-")
#     with open(file_name,'wb+') as f:
#         f.write(image.file.read())
#         f.close()
#     img=cv2.imread(file_name)
#     image_predict = get_predict(model,face_detector,img)
#     cv2.imwrite("1.png",img)
#     return {"filename": file}
@app.post("/predict")
async def get_segmentation_map(file: UploadFile = File(...)):
    """Get segmentation maps from image file"""
    image_predict = get_predict(model,face_detector,await file.read())
    bytes_io = io.BytesIO()
    image_predict.save(bytes_io, format="PNG")
    return Response(bytes_io.getvalue(), media_type="image/png")
