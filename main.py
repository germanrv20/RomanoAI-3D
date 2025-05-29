import cv2
import os
import pathlib
import numpy as np
import cuia
import math


# Cargar modelos
red = cv2.dnn.readNetFromCaffe("dnn/deploy.prototxt", "dnn/res10_300x300_ssd_iter_140000.caffemodel")
fr = cv2.FaceRecognizerSF.create("dnn/face_recognition_sface_2021dec.onnx", "")
detector_caras  = cv2.FaceDetectorYN.create("dnn/face_detection_yunet_2023mar.onnx", config="", input_size=(320, 320), score_threshold=0.7)

# Base de datos de usuarios con información adicional
usuarios = {
    "German": {"pais": "España", "idioma": "Español", "vector": None},
    "Zakaria": {"pais": "Marruecos", "idioma": "Árabe", "vector": None},
    "Angelina Jolie": {"pais": "EE.UU.", "idioma": "Inglés", "vector": None},
    "Brad Pitt": {"pais": "EE.UU.", "idioma": "Inglés", "vector": None},
    "PabloGolfo": {"pais": "España.", "idioma": "Español", "vector": None},
    # Añadir más usuarios según sea necesario
}

# ID del marcador : archivo del modelo GLB 
modelos_por_id = { 
    0: "media/templo_de_saturno_roma.glb", 
    1: "media/foro-romano.glb", 
    2: "media/libarna_-_anfiteatro.glb", 
    3: "media/libarna_-_foro.glb", 
    4: "media/modelo1.glb", 
    5: "media/modelo2.glb", 
    6: "media/modelo3.glb", 
    7: "media/modelo4.glb", 
    8 :"media/tierra.glb",
  # Agrega más IDs y modelos aquí 
  }

# Cargar imágenes de Usuarios y obtener vectores
Usuarios = []
lista = os.listdir("media/Usuarios/")
for f in lista:
    ruta = os.path.join("media/Usuarios", f)
    nombre = pathlib.Path(f).stem    
    img = cv2.imread(ruta)
    if img is None:
        print(f"⚠️ No se pudo cargar la imagen {ruta}")
        continue

    h, w, _ = img.shape
    detector_caras .setInputSize((w, h))
    ret, caranueva = detector_caras .detect(img)
    if ret and caranueva is not None:
        caracrop = fr.alignCrop(img, caranueva[0])
        codcara = fr.feature(caracrop)
        if nombre in usuarios:
            usuarios[nombre]["vector"] = codcara
            usuarios[nombre]["imagen"] = img.copy()  # ✅ Añadir esta línea
            print(f"✅ Cargado vector de {nombre}")
    else:
        print(f"❌ No se detectó cara en {nombre}")

# Iniciar webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow (más estable en Windows)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    exit()

# Leer primer frame para ajustar el tamaño del detector
ret, frame = cap.read()
if not ret:
    print("❌ No se pudo leer la imagen de la cámara")
    cap.release()
    exit()

h, w, _ = frame.shape
detector_caras .setInputSize((w, h))

# ===============================
# CONFIGURACIÓN INICIAL
# ===============================
myCam = 0  # Índice de la cámara a utilizar
bestCap = cuia.bestBackend(myCam)  # Seleccionamos el mejor backend disponible para OpenCV

# ===============================
# CARGA DEL DICCIONARIO Y DETECTOR ARUCO
# ===============================
diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
detector_aruco = cv2.aruco.ArucoDetector(diccionario)

# ===============================
# DEFINICIÓN DE FUNCIONES AUXILIARES
# ===============================


def origen(TAM):
    return np.array([[-TAM/2.0, -TAM/2.0, 0.0],
                     [-TAM/2.0,  TAM/2.0, 0.0],
                     [ TAM/2.0,  TAM/2.0, 0.0],
                     [ TAM/2.0, -TAM/2.0, 0.0]])

def proyeccion(puntos, rvec, tvec, cameraMatrix, distCoeffs):
    puntos = np.array(puntos, dtype=np.float32)
    proyectados, _ = cv2.projectPoints(puntos, rvec, tvec, cameraMatrix, distCoeffs)
    return [tuple(map(int, p[0])) for p in proyectados]

def fov(cameraMatrix, ancho, alto):
    if ancho > alto:
        f = cameraMatrix[1, 1]
        fov_rad = 2 * np.arctan(alto / (2 * f))
    else:
        f = cameraMatrix[0, 0]
        fov_rad = 2 * np.arctan(ancho / (2 * f))
    return np.rad2deg(fov_rad)

def fromOpencvToPygfx(rvec, tvec):
    pose = np.eye(4)
    pose[0:3, 3] = tvec.T
    pose[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
    pose[1:3] *= -1
    pose = np.linalg.inv(pose)
    return pose

def detectarPose(frame, tam):
    bboxs, ids, _ = detector_aruco.detectMarkers(frame)
    if ids is not None:
        objPoints = np.array([[-tam/2.0, tam/2.0, 0.0],
                              [tam/2.0, tam/2.0, 0.0],
                              [tam/2.0, -tam/2.0, 0.0],
                              [-tam/2.0, -tam/2.0, 0.0]])
        resultado = {}
        for i in range(len(ids)):
            ret, rvec, tvec = cv2.solvePnP(objPoints, bboxs[i], cameraMatrix, distCoeffs)
            if ret:
                resultado[ids[i][0]] = (rvec, tvec)
        return (True, resultado)
    return (False, None)

"""
def realidadMixta(frame):
    ret, pose = detectarPose(frame, 0.19)
    if ret and pose[0]:
        M = fromOpencvToPygfx(pose[0][0], pose[0][1])
        escena.actualizar_camara(M)
        imagen_render = escena.render()
        imagen_render_bgr = cv2.cvtColor(imagen_render, cv2.COLOR_RGBA2BGRA)
        resultado = cuia.alphaBlending(imagen_render_bgr, frame)
    else:
        resultado = frame
    return resultado
"""



def realidadMixta(frame):
    ret, poses = detectarPose(frame, 0.19)
    resultado = frame

    if ret and poses:
        imagen_render = np.zeros((alto, ancho, 4), dtype=np.uint8)  # RGBA acumulador

        for id_detectado, (rvec, tvec) in poses.items():
            if id_detectado in modelos and id_detectado in escenas:
                modelo = modelos[id_detectado]
                escena = escenas[id_detectado]

                M = fromOpencvToPygfx(rvec, tvec)
                escena.actualizar_camara(M)

                imagen_modelo = escena.render()
                imagen_modelo_bgr = cv2.cvtColor(imagen_modelo, cv2.COLOR_RGBA2BGRA)

                # Combinar el modelo renderizado sobre la acumulada
                imagen_render = cuia.alphaBlending(imagen_modelo_bgr, imagen_render)

        # Finalmente, combinar todo con el frame original
        resultado = cuia.alphaBlending(imagen_render, frame)

    return resultado






# ===============================
# PARÁMETROS DE CÁMARA
# ===============================
webcam = cv2.VideoCapture(myCam, bestCap)
ancho = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
webcam.release()
try:
    import camara
    cameraMatrix = camara.cameraMatrix
    distCoeffs = camara.distCoeffs
except ImportError:
    cameraMatrix = np.array([[1000, 0, ancho/2],
                             [0, 1000, alto/2],
                             [0, 0, 1]])
    distCoeffs = np.zeros((5, 1))

modelos = {}
escenas = {}  # Nueva: una escena por modelo
for id_aruco, ruta in modelos_por_id.items():
    m = cuia.modeloGLTF(ruta)
    m.rotar((math.pi/2.0, 0, 0))
    m.escalar(0.15)
    m.flotar()
    anims = m.animaciones()
    if len(anims) > 0:
        m.animar(anims[0])

    escena = cuia.escenaPYGFX(fov(cameraMatrix, ancho, alto), ancho, alto)
    escena.agregar_modelo(m)
    escena.ilumina_modelo(m)
    escena.iluminar()

    modelos[id_aruco] = m
    escenas[id_aruco] = escena  # Guardamos escena individual
# ===============================
# INICIALIZAR VIDEO Y EJECUTAR
# ===============================
ar = cuia.myVideo(myCam, bestCap)
ar.process = realidadMixta

# ===============================
# FLUJO PRINCIPAL
# ===============================
usuario_reconocido = False
nombre_reconocido = ""
maximo_reconocido = 0
tiempo_reconocido = None  # Para registrar el tiempo de primer reconocimiento

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Si el usuario ha sido reconocido
    if usuario_reconocido:
        import time

        # Si es la primera vez que se reconoce al usuario, inicializamos el tiempo
        if tiempo_reconocido is None:
            tiempo_reconocido = time.time()  # Registramos el tiempo en el primer reconocimiento

        tiempo_actual = time.time()

        # Mostrar "Bienvenido" solo durante los primeros 3 segundos
        if tiempo_actual - tiempo_reconocido <= 3:
            texto = f"Bienvenido {nombre_reconocido} ({maximo_reconocido:.2f})"
            cv2.putText(frame, texto, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Obtener la imagen del usuario y crear la máscara circular
        img_user = usuarios.get(nombre_reconocido, {}).get("imagen", None)
        if img_user is not None:
            # Redimensionar a tamaño más pequeño (70x70)
            thumb = cv2.resize(img_user, (70, 70))

            # Crear máscara circular
            mask = np.zeros((70, 70), dtype=np.uint8)
            cv2.circle(mask, (35, 35), 35, 255, -1)

            # Aplicar máscara
            thumb_circle = cv2.bitwise_and(thumb, thumb, mask=mask)

            # Coordenadas para esquina superior derecha
            h_frame, w_frame, _ = frame.shape
            x_offset = w_frame - 80
            y_offset = 30

            # Nombre justo encima
            cv2.putText(frame, f"{nombre_reconocido}", (x_offset , y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Mostrar imagen circular
            roi = frame[y_offset:y_offset+70, x_offset:x_offset+70]
            roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
            final = cv2.add(roi_bg, thumb_circle)
            frame[y_offset:y_offset+70, x_offset:x_offset+70] = final

        # Ahora aplicar la realidad mixta al frame
        frame_rm = realidadMixta(frame)

        # Mostrar el frame con realidad mixta
        cv2.imshow("Bienvenido", frame_rm)

    else:
        # Si el usuario no está reconocido, mostrar el frame normal
        cv2.imshow("¿Eres uno de los usuarios?", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc para salir
        break

        


    # Detectar caras
    ret, caras = detector_caras .detect(frame)
    if ret and caras is not None:
        for cara in caras:
            c = cara.astype(int)
            try:
                caracrop = fr.alignCrop(frame, cara)
                codcara = fr.feature(caracrop)
            except:
                continue

            maximo = -999
            nombre = "Desconocido"

            for usuario, info in usuarios.items():
                if info["vector"] is not None:
                    semejanza = fr.match(info["vector"], codcara, cv2.FaceRecognizerSF_FR_COSINE)
                    if semejanza > maximo:
                        maximo = semejanza
                        nombre = usuario

            if maximo < 0.4:
                color = (0, 0, 255)
                texto = "Desconocido"
            else:
                color = (0, 255, 0)
                texto = f"Bienvenido {nombre} ({maximo:.2f})"

                if nombre != "Desconocido":
                    info = usuarios[nombre]
                    cv2.putText(frame, f"Pais: {info['pais']}", (c[0], c[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, f"Idioma: {info['idioma']}", (c[0], c[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                usuario_reconocido = True
                nombre_reconocido = nombre
                maximo_reconocido = maximo

            cv2.rectangle(frame, (c[0], c[1]), (c[0]+c[2], c[1]+c[3]), color, 3)
            cv2.putText(frame, texto, (c[0], c[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("¿Eres uno de los usuarios?", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc para salir
        break

cap.release()
cv2.destroyAllWindows()
