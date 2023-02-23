import os
import cv2

pad = r"C:\Users\Hendrik.Clijsters\MOSA-RT\OneDrive - MOSA-RT\Prog\PROG_6IICT\_6IICT_PROG4_opl\hfst_5_AI\objectdetectie\toiletpapier_hendrik" # pad naar de map verwerken
folder = os.listdir(pad)
for index, bestand in enumerate(folder):
    # Sla bestanden over die geen afbeeldingen zijn
    if (".jpg" not in bestand):
        continue
    # Bepaal naam zonder extensie + Inladen in Python
    naam = bestand.replace(".jpg","")
    afbeelding = cv2.imread(f'{pad}/{bestand}')

    # Roteer afbeelding en sla op
    foto_rot = cv2.rotate(afbeelding, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(filename=f'{pad}/{naam}_rot.jpg', img=foto_rot)

    # VUL AAN voor spiegel + filter
    foto_spiegel = cv2.flip(afbeelding, -1)
    cv2.imwrite(filename=f'{pad}/{naam}_spiegel.jpg', img=foto_spiegel)

    foto_filter = cv2.GaussianBlur(afbeelding, (35,35), cv2.BORDER_DEFAULT)
    cv2.imwrite(filename=f'{pad}/{naam}_filter.jpg', img=foto_filter)