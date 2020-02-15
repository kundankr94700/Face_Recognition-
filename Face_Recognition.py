import cv2
import numpy as np
from os import listdir,mkdir
from os.path import isfile, join
from tkinter import *
import pyttsx3 as t2s
from tkinter.font import  Font
from csv import *
import threading
from PIL import ImageTk, Image
face_cascade = cv2.CascadeClassifier('face_cas.xml')
def Recognise_Photo():
    try:

        eng2 = t2s.init()
        models=[]
        def training(data_path):
            onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]


            Training_Data, Labels = [], []

            for i, files in enumerate(onlyfiles):
                image_path = data_path + onlyfiles[i]
                images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                Training_Data.append(np.asarray(images, dtype=np.uint8))
                Labels.append(i)

            Labels = np.asarray(Labels, dtype=np.int32)
            model = cv2.face.LBPHFaceRecognizer_create()

            model.train(np.asarray(Training_Data), np.asarray(Labels))
            return model

        onlyfiles1 = [f for f in listdir('D:/Face_Recognition/Camera/')]
        for i in range((len(onlyfiles1)//2)+1):
            try:
                models.append(training('D:/Face_Recognition/Camera/Face_Data_%d/'%(i+1)))
            except:
                pass

        model = models[0]
        pre = 0
        csv_file = 'D:/Face_Recognition/Camera/Face_Record 1.csv'

        def text2speech_name(Id):
            try:
                eng2.setProperty('rate', 90)
                eng2.setProperty('volume', .9)
                eng2.say(Id)
                eng2.runAndWait()
            except:
                pass

        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (200, 200))
                j, k = model.predict(face)
                confidence = int(100 * (1 - (k) / 300))

                ref1, ref2 = j // 20, k // 10

                if (ref2 < 5):
                    if confidence > 75:
                        Id = 'Searching'
                        f1 = open(csv_file, 'r')
                        r = reader(f1)
                        for ii in r:

                            try:

                                if ref1 == int(ii[0]):
                                    Id = ii[1]

                                    break
                            except:
                                pass

                        cv2.putText(frame, str(Id), (x + w, y + h), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 245, 0), 1)
                        try:
                            t = threading.Thread(name='child', target=text2speech_name, args=(Id,))
                            if not t.is_alive():
                                t.start()
                        except:
                            pass

                    else:
                        pass
                else:
                    if pre <(len(onlyfiles1)//2) -1 :
                        try:
                            model = models[pre+1]
                            csv_file = 'D:/Face_Recognition/Camera/Face_Record %d.csv'%(pre+2)

                        except:
                            pass
                        pre = pre+1

                    elif pre == (len(onlyfiles1)//2)-1:
                        try:
                            model = models[0]
                            csv_file = 'D:/Face_Recognition/Camera/Face_Record 1.csv'

                        except:
                            pass
                        pre = 0

                    else:
                        pass
            cv2.imshow('Face Recognition Software', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except:
        pass

def add_face():
    try:
        try:
            mkdir('D:/Face_Recognition/Camera/')
        except:
            pass

        def get_name():
            root_add.destroy()
            x112 = s11.get()

            def process_add(data_path, point):
                global po
                onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
                Labels = []
                for i, files in enumerate(onlyfiles):
                    Labels.append(i)
                Labels = np.asarray(Labels, dtype=np.int32)
                x2 = len(Labels)
                x1 = len(Labels) // 20
                name = x112

                f = open('D:/Face_Recognition/Camera/Face_Record %d.csv' % (point), 'a')
                w = writer(f)
                w.writerows([[x1, name]])
                f.close()
                cap = cv2.VideoCapture(0)
                count = 0

                while True:
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    n = 0
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                        n = faces.shape[0]
                        face_img = frame[y:y + h, x:x + w]
                    if n == 0:
                        cv2.putText(frame, 'No Face found', (10, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                    else:
                        count += 1
                        face = cv2.resize(face_img, (200, 200))
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite('%s/user' % data_path + str(count + x2) + '.jpg', face)
                        cv2.putText(face, str(count), (5, 180), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Face ', face)
                    cv2.imshow('Capture Face_Data', frame)
                    if cv2.waitKey(1) == ord('q') or count == 20:
                        break

                count = 0
                cap.release()
                cv2.destroyAllWindows()


            data_path = 'D:/Face_Recognition/Camera/'
            onlyfiles1 = [f for f in listdir(data_path)]
            xx = len(onlyfiles1) // 2 + 1
            try:
                mkdir('D:/Face_Recognition/Camera/Face_Data_%d/' % (xx))
            except:
                pass

            process_add('D:/Face_Recognition/Camera/Face_Data_%d/' % (xx), xx)
            #print("photo Added")

        root_add = Toplevel()
        root_add.geometry('430x450+950+100')
        img1 = ImageTk.PhotoImage(Image.open("123.png"))
        panel = Label(root_add, image=img1).place(x=60, y=10)
        l1 = Label(root_add, text='Face Recognition Software', font=f5, fg='Red').place(x=80, y=5)
        l1 = Label(root_add, text='Enter Name of the Person ',font=f5,fg='blue').place(x=80, y=330)
        s11 = StringVar()
        ent = Entry(root_add, textvariable=s11,font=f5).place(x=100, y=370)
        b = Button(root_add, text='Add Person Record', command=get_name, width=25, bg='Green', height=1,fg='white',
                   font=f3).place(x=80, y=410)
        c1 = Canvas(root_add, width=10, height=750, bg='Brown')
        c1.pack(side=RIGHT)
        c1 = Canvas(root_add, width=10, height=750, bg='Brown')
        c1.pack(side=LEFT)
        root_add.resizable('false', 'false')
        root_add.mainloop()
    except:
        pass

try:
    mkdir('D:/Face_Recognition/')
except:
    pass
root_Face_Reognise = Tk()
root_Face_Reognise.geometry('750x420+600+100')
frame=Frame(root_Face_Reognise,height=600,width=10).pack(side=RIGHT)
img = ImageTk.PhotoImage(Image.open("face_reg.jpg"))
panel = Label(frame, image = img)
panel.pack(side = "left", fill = "both", expand = "yes")
f5 = Font(family="Time New Roman", size=15, weight="bold")
root_Face_Reognise.title('Face Recognition Software')
f1 = Font(family="Time New Roman", size=16, weight="bold", underline=1)
f3 = Font(family="Time New Roman", size=12, weight="bold")
l3 = Label(root_Face_Reognise, text="Machine Learning Training Project", fg='Black', font=f1).place(x=10, y=30)
l = Label(root_Face_Reognise, text='Face Recogniton Software', fg='red', font=f1).place(x=40, y=80)
b1 = Button(root_Face_Reognise, text='Add Face Data', bg='skyBlue', fg='white', width=25, height=1, command=add_face,
            font=f5).place(x=30, y=210)
b1 = Button(root_Face_Reognise, text='Recognise', bg='black', fg='white', width=25, height=1,command=Recognise_Photo, font=f5).place(x=30, y=270)

b1 = Button(root_Face_Reognise, text='Close', bg='red', width=25, height=1, command=root_Face_Reognise.destroy,
            font=f5).place(x=30, y=330)
root_Face_Reognise.resizable(False, False)
root_Face_Reognise.mainloop()
