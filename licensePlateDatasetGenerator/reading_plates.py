import os
import cv2
import pytesseract



os.chdir("data")
plates = os.listdir()
print("---------------------------------")
for plate in plates:
    plate_number = cv2.imread(plate)
    text = pytesseract.image_to_string(plate_number)
    cv2.ShowImage("plate_number",plate_number)
    cv2.waitKey(200)
    print("real plate: "+plate[:-4])
    print("plate read:"+text)
    print("---------------------------------")
os.chdir("..")