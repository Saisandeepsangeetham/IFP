import os

os.chdir("D:/IFP/Face_Detect/dataset")
print(os.getcwd())

face_user = []
for dirpath,dir,filename in os.walk(os.getcwd()):
    for i in filename:
        face_user.append()