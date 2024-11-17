import mediapipe as mp
import cv2 as cv 
import numpy as np
mp_face_mesh = mp.solutions.face_mesh
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
# Left eye indices list
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# Right eye indices list
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]


cap = cv.VideoCapture(0)
with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            # print(results.multi_face_landmarks[0].landmark)
            mesh_points = (np.array([np.multiply([p.x,p.y], [img_w,img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark]))
            # cv.polylines(frame, [mesh_points[LEFT_IRIS]],True,(255,0,0),thickness=1,lineType=cv.LINE_AA)
            # cv.polylines(frame, [mesh_points[RIGHT_IRIS]],True,(0,255,0),thickness=1,lineType=cv.LINE_AA)
            (l_cx, l_cy), radius_l = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), radius_r = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_l = (int(l_cx),int(l_cy))
            center_r = (int(r_cx),int(r_cy))
            cv.circle(frame, center_l, int(radius_l), (0,255,0), thickness=1, lineType=cv.LINE_AA)
            cv.circle(frame, center_r, int(radius_r), (0,0,255), thickness=1, lineType=cv.LINE_AA)
            cv.imshow('out',frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            





        