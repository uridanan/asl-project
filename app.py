import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import tensorflow as tf
import numpy as np
from collections import deque, Counter
import os
import json
import base64
import urllib.request

def get_ice_servers():
    # Always include Google STUN server by default
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    
    twilio_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    twilio_token = os.environ.get("TWILIO_AUTH_TOKEN")
    metered_key = os.environ.get("METERED_API_KEY")
    turn_url = os.environ.get("TURN_SERVER_URL")
    turn_username = os.environ.get("TURN_SERVER_USERNAME")
    turn_credential = os.environ.get("TURN_SERVER_CREDENTIAL")
    
    # Safely try fetching from streamlit secrets if env vars aren't present
    try:
        if not twilio_sid and "TWILIO_ACCOUNT_SID" in st.secrets:
            twilio_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        if not twilio_token and "TWILIO_AUTH_TOKEN" in st.secrets:
            twilio_token = st.secrets["TWILIO_AUTH_TOKEN"]
        if not metered_key and "METERED_API_KEY" in st.secrets:
            metered_key = st.secrets["METERED_API_KEY"]
        if not turn_url and "TURN_SERVER_URL" in st.secrets:
            turn_url = st.secrets["TURN_SERVER_URL"]
        if not turn_username and "TURN_SERVER_USERNAME" in st.secrets:
            turn_username = st.secrets["TURN_SERVER_USERNAME"]
        if not turn_credential and "TURN_SERVER_CREDENTIAL" in st.secrets:
            turn_credential = st.secrets["TURN_SERVER_CREDENTIAL"]
    except Exception:
        pass

    # 1. Fetch Twilio TURN Servers if credentials are provided
    if twilio_sid and twilio_token:
        try:
            url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_sid}/Tokens.json"
            req = urllib.request.Request(url, method="POST")
            auth_str = f"{twilio_sid}:{twilio_token}"
            auth_b64 = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
            req.add_header("Authorization", f"Basic {auth_b64}")
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
                twilio_servers = data.get("ice_servers", [])
                if twilio_servers:
                    return twilio_servers
        except Exception as e:
            st.sidebar.warning(f"Note: Twilio auth failed. Falling back to default configuration.")

    # 2. Fetch Metered (Open Relay) TURN Servers if API key is provided
    if metered_key:
        try:
            url = f"https://openrelay.metered.ca/api/v1/turn/credentials?apiKey={metered_key}"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                metered_servers = json.loads(response.read().decode("utf-8"))
                if metered_servers:
                    return metered_servers
        except Exception as e:
            st.sidebar.warning(f"Note: Metered auth failed. Falling back to default configuration.")

    # 3. Add manual TURN server if explicitly provided
    if turn_url:
        manual_server = {"urls": [turn_url]}
        if turn_username:
            manual_server["username"] = turn_username
        if turn_credential:
            manual_server["credential"] = turn_credential
        ice_servers.append(manual_server)

    return ice_servers


# 1. הגדרות דף האפליקציה
# פונקציה שקובעת את כותרת הטאב בדפדפן ואת פריסת הדף (רחבה)
st.set_page_config(page_title="ASL Real-Time Translator", layout="wide")

# הוספת עיצוב CSS מותאם אישית כדי לשפר את המראה הוויזואלי של האפליקציה
st.markdown("""
   <style>
   .main { background-color: #f5f7f9; }
   .ststInfo { background-color: #ffffff; border-radius: 10px; padding: 20px; border: 1px solid #e0e0e0; }
   h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
   </style>
   """, unsafe_allow_html=True)

# 2. תפריט צד (Sidebar) וטעינת המודל
st.sidebar.title("Configuration & Guide")
st.sidebar.info("This app uses a CNN model to translate ASL fingerspelling into English text in real-time.")

try:
    st.sidebar.image("guide.png", caption="ASL Fingerspelling Guide", use_container_width=True)
except:
    st.sidebar.warning("Note: Place 'guide.png' in your project folder.")

# פונקציה לטעינת המודל עם Cache (זיכרון מטמון)
# השימוש ב-cache_resource מבטיח שהמודל ייטען רק פעם אחת ולא בכל רענון של הדף
@st.cache_resource
def load_my_model():
    from load_model_compat import load_model_compatible
    return load_model_compatible("asl_recognition_new_model.keras")

model = load_my_model()

# רשימת המחלקות (האותיות) שהמודל יודע לזהות, לפי הסדר שבו אומן
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


# 3. מחלקת עיבוד הווידאו (הלב של המערכת)
# מחלקה זו מטפלת בכל פריים שמגיע מהמצלמה ומבצעת עליו את החיזוי
class ASLVideoProcessor:
    def __init__(self):
        # buffer: תור (Queue) ששומר את 10 החיזויים האחרונים כדי ליצור יציבות בזיהוי
        self.buffer = deque(maxlen=10)
        # last_added_letter: שומר את האות האחרונה שנוספה לטקסט כדי למנוע כפילויות
        self.last_added_letter = None
        # frame_count: מונה פריימים לצורך דילוג על פריימים (אופטימיזציה)
        self.frame_count = 0
        # final_text: המחרוזת הסופית של המילים שהמשתמש כתב
        self.final_text = ""
        # display_label: הטקסט שמוצג מעל תיבת המצלמה (האות הנוכחית)
        self.display_label = "Scanning..."

    # פונקציית recv נקראת באופן אוטומטי עבור כל פריים וידאו חדש
    def recv(self, frame):
        # המרת הפריים למערך NumPy (פורמט ש-OpenCV מבין)
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # היפוך מראה לנוחות המשתמש
        h, w, _ = img.shape
        self.frame_count += 1

        # הגדרת אזור העניין (ROI) - הריבוע הירוק שבו המשתמש שם את היד
        x1, y1, x2, y2 = 100, 100, 350, 350
        roi = img[y1:y2, x1:x2]

        # דילוג מינימלי על פריימים (3) כדי לשמור על מהירות וגם על איכות הסיווג
        if self.frame_count % 3 == 0:
            # עיבוד מקדים: שינוי גודל ל-64x64 והמרה ל-RGB (כפי שהמודל אומן)
            processed = cv2.resize(roi, (64, 64))
            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(rgb.astype("float32"), axis=0)

            # הרצת החיזוי דרך המודל
            prediction = model.predict(input_data, verbose=0)
            class_id = np.argmax(prediction)
            confidence = prediction[0][class_id]

            # החלטה על האות בהתאם לסף הביטחון (0.3)
            letter = class_names[class_id] if confidence > 0.30 else "nothing"

            # לוגיקה של הוספת אותיות למשפט (רק אם הביטחון מעל 0.7)
            if letter != "nothing" and confidence > 0.70:
                self.buffer.append(letter)
                # בדיקה אם יש הסכמה ב-buffer (לפחות 8 מתוך 10 פריימים זהים)
                if len(self.buffer) == 10:
                    most_common, count = Counter(self.buffer).most_common(1)[0]
                    # טיפול במקרים מיוחדים: רווח ומחיקה
                    if count >= 8 and most_common != self.last_added_letter:
                        if most_common == "space":
                            self.final_text += " "
                        elif most_common == "del":
                            self.final_text = self.final_text[:-1]
                        else:
                            self.final_text += most_common
                        self.last_added_letter = most_common
                        self.buffer.clear()

            if letter == "nothing":
                self.last_added_letter = None

            # עדכון התווית שתוצג על המסך
            self.display_label = f"{letter} ({confidence:.2f})"

        # ציור הממשק בתוך הווידאו
        cv2.rectangle(img, (x1, y1), (x2, y2), (46, 204, 113), 2)
        cv2.putText(img, self.display_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (46, 204, 113), 2)

        # ציור רקע שחור שקוף בתחתית המסך להצגת התוצאה
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        # הצגת המשפט המלא שנכתב עד כה
        cv2.putText(img, f"Result: {self.final_text}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # החזרת הפריים המעובד לממשק ה-Streamlit
        return frame.from_ndarray(img, format="bgr24")


# 4. Main UI Layout
st.title("🤟 ASL Real-Time Translator")
st.write("Hold your hand signs within the green box to start translating")

webrtc_streamer(
    key="asl-translator",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": get_ice_servers()}),
    video_processor_factory=ASLVideoProcessor,
    async_processing=True,
)

st.markdown("---")
st.caption("Developed with ❤️ for Machin3e Learning Project 2026- Lielle Danan")