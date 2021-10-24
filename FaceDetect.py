import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
faceMesh = mp_face_mesh.FaceMesh()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    resultat = faceMesh.process(img)

    print(resultat.multi_face_landmarks)

    if resultat.multi_face_landmarks:
        for face_landmarks in resultat.multi_face_landmarks:
            mp_draw.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION
                                   , mp_draw.DrawingSpec((0, 255, 0), 1, 1),
                                   mp_draw.DrawingSpec((0, 0, 255), 1, 1))






    cv2.imshow("face detect", img)
    cv2.waitKey(1)
