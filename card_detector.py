
import numpy as np
import cv2
import model.model as m
import config.model_config as mc
import config.config as cfg

dl_model = m.yolo_model.load_from_file(mc.model_filename, cfg.output_shape)

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('p'):

        np_im = np.array(frame)
        np_im = np_im[np.newaxis, ...]
        print('predicting...')
        prediction = dl_model.predict(np_im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

