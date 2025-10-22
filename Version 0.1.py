import cv2
import mediapipe as mp
import numpy as np
mp_face_mesh=mp.solutions.face_mesh #solutions is submodule inside mediapipe that contains pre-built ML solutions, it contains read-made classes and functions
'''mp.solutions.face_mesh -> Face Mesh class
mp.solutions.hands -> Hand tracking class
mp.solutions.pose -> Full body pose estimation class'''
face_mesh=mp_face_mesh.FaceMesh(static_image_mode=False,
                                max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)
#face_mesh holds a FaceMesh object which is an instance of face_mesh class
#when mp_face_mesh.FaceMesh() function is called, python instantiates an object, allocates memory for it, and loads the underlyind model
'''static_mode_image = False -> if True face detection runs every time, if false, detected once during first frame
and track wherever landmarks move
Keep true if using multiple images, and false for real-time video tracking'''
'''max_num_faces=1 -> detect a primary face'''
'''refine_landmarks=True -> Refines accuracy of detected landmarks by adding extra landmarks for eyes and lips
beyond 468 base landmarks
high-precision landmarks around facial features
micro-expressions accurately captured
Adds 20–30 extra points per eye and 20–30 points around lips
Uses higher-resolution heat-maps around critical features
reduced landmark jitter during blinking, lip movement'''
'''min_detection_confidence=0.5 -> detector will ignore faces with confidence < 0.5'''
'''min_tracking_confidence -> Threshold for tracking confidence between frames, tracker predict landmarks based
on previous frames, if confidence<0.5 MediaPipe will redo detection instead of tracking'''
mp_draw=mp.solutions.drawing_utils
#drawing_utils is a module inside MP, -> to draw landmarks and connections
'''Mediapipe's face mesh landmarks outputs as (x,y,z), landmark.x is normalised horizontal position, and so on
for y and z, these are not images yet, in order to visualise, we need to draw
->points for each landmark
->lines connecting landmarks forming a mesh
draw_utils provides
    ->draw_landmarks() - draw points and lines on image
    ->DrawingSpec - specify color, thickness, radius for points and lines
Normal coordinates -> pixel coordinates -> draw circles using DrawingSpec, lines using connections parameter
'''
drawing_spec=mp_draw.DrawingSpec(color=(255,255,255),thickness=1,circle_radius=1)
#DrawingSpec is a class inside mp_draw
#we are creating a drawing_spec object
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        break
    frame=cv2.flip(frame,1) #to avoid mirror effect
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #mp models were trained using RGB images, hence swapping B and R can make detection less accurate
    results=face_mesh.process(rgb)
    #.process is the method of FaceMesh class, this feeds a frame to the FaceMesh model
    '''1) Preprocessing - Resizes img to 192x192 or 256x256 depending on model
                                   Normalises pixel values
                                   Converts image into tensor the model can process
    2) Face detection - detection or tracking as per user want
                                 return bounding boxes and confidence scores internally
    3) Landmark prediction - 468 or more(if refine_landmarks=True) 3D landmarks are predicted
                                            Each landmark has:
                                                landmark.x (0-1)
                                                landmark.y (0-1)
                                                landmark.z (-ve in front of cam)
                                            Normalized means: (0,0)->top-left, (1,1)->bottom right
    4) Tracking(if video)
    5) Packaging results - Returns a NamedTuple object
                    results.multi_face_landmarks ->List of NormalizedLandmarkList object
                                                                      Each NormalizedLandmarkList has 468 landmarks
                                            results.multi_face_landmarks[0].landmark[0].x  # first landmark x-coordinate
                                            results.multi_face_landmarks[0].landmark[0].y  # first landmark y-coordinate
                                            results.multi_face_landmarks[0].landmark[0].z  # first landmark depth

                    results.multi_face_world_landmarks  # 3D landmarks in real-world coordinates instead of normalized 0-1
                    results.multi_face_geometry        # additional geometry info (used for AR pipelines)
    '''
    mesh_canvas=np.zeros_like(frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(image=mesh_canvas,
                                   landmark_list=face_landmarks,
                                   connections=mp_face_mesh.FACEMESH_TESSELATION,
                                   landmark_drawing_spec=drawing_spec,
                                   connection_drawing_spec=drawing_spec)
    combined=np.hstack((frame,mesh_canvas))
    #Horizontally stack arrays, creates split-screen view, concatenates arrays along 2nd dimension(width)
    cv2.imshow("Face Mesh",combined)
    if cv2.waitKey(1) & 0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
