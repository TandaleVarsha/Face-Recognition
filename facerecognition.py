from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.clock import Clock
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

class CamApp(App):
    def build(self):
        self.web_cam = KivyImage(size_hint=(1, .8))
        self.status_label = Label(text="Verification Uninitiated", size_hint=(1, .1), color=(1, 1, 1, 1))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1, .1))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.status_label)
        layout.add_widget(self.button)

        self.button.background_color = (0.2, 0.2, 0.8, 1)

        self.model = tf.keras.models.load_model('siamesemodelv2.h5', custom_objects={'L1Dist': L1Dist})
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return layout

    def update(self, *args):
        ret, frame = self.capture.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]

        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic_model.process(image)
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check if face landmarks are detected
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(
                    color=(255, 0, 255),
                    thickness=1,
                    circle_radius=1
                ),
                mp_drawing.DrawingSpec(
                    color=(0, 255, 255),
                    thickness=1,
                    circle_radius=1
                )
            )
            self.status_label.text = "Verification Uninitiated"
        else:
            #self.status_label.text = "Face Not Detected"
            cv2.putText(image, "Face Not Detected", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Convert the image to texture and update the Kivy Image
        buf = cv2.flip(image, 0).tostring()
        img_texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)

        img = tf.image.resize(img, (100, 100))
        img = img / 255.0

        return img

    def verify(self, *args):
        detection_threshold = 0.40
        verification_threshold = 0.2

        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]
        cv2.imwrite(SAVE_PATH, frame)

        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))

            result = self.model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
            results.append(result)

        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold

        self.status_label.text = 'Face Recognized' if verified else 'Unknown Face'

        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        return results, verified

if __name__ == '__main__':
    Window.size = (500, 500)
    CamApp().run()
