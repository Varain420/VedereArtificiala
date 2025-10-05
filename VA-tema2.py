# Laborator 2 - Detecția de forme, Histograme
# Pentru a rula acest script, trebuie să aveți instalate următoarele biblioteci:
# pip install opencv-python numpy matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os


# --- Funcție ajutătoare pentru a descărca și încărca o imagine de test ---
def incarcare_imagine_test(url, nume_fisier='imagine_test.jpg'):
    """Descarcă o imagine de la un URL și o încarcă folosind OpenCV."""
    try:
        urllib.request.urlretrieve(url, nume_fisier)
        img = cv2.imread(nume_fisier)
        if img is None:
            raise Exception("Imaginea nu a putut fi încărcată. Verificați calea sau formatul fișierului.")
        # OpenCV încarcă imaginile în format BGR, le convertim în RGB pentru afișare corectă cu Matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception as e:
        print(f"Eroare la încărcarea imaginii: {e}")
        # Generează o imagine neagră ca fallback în caz de eroare
        return np.zeros((400, 400, 3), dtype=np.uint8)


# --- Funcție ajutătoare pentru a afișa imagini ---
def afiseaza_imagini(imagini, titluri, titlu_principal):
    """Afișează o listă de imagini folosind Matplotlib."""
    numar_imagini = len(imagini)
    plt.figure(figsize=(15, 5))
    plt.suptitle(titlu_principal, fontsize=16)
    for i in range(numar_imagini):
        plt.subplot(1, numar_imagini, i + 1)
        plt.imshow(imagini[i], cmap='gray' if len(imagini[i].shape) == 2 else None)
        plt.title(titluri[i])
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- Sarcina 1: Detectează cercuri într-un video ---
def sarcina_1(cale_video_intrare, cale_video_iesire='v1_output.mp4'):
    """
    Citește un fișier video, detectează cercurile în fiecare cadru și
    salvează rezultatul într-un nou fișier video.
    """
    print("--- Sarcina 1: Detecție cercuri în video ---")
    if not os.path.exists(cale_video_intrare):
        print(f"Eroare: Fișierul video '{cale_video_intrare}' nu a fost găsit.")
        print("Vă rugăm să descărcați fișierul 'v1.mp4' sau să specificați o altă cale.")
        return

    # Deschidem fișierul video
    cap = cv2.VideoCapture(cale_video_intrare)
    if not cap.isOpened():
        print("Eroare la deschiderea fișierului video.")
        return

    # Obținem proprietățile video pentru a crea fișierul de ieșire
    latime_cadru = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    inaltime_cadru = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Definim codec-ul și creăm obiectul VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Sau 'XVID' pentru .avi
    out = cv2.VideoWriter(cale_video_iesire, fourcc, fps, (latime_cadru, inaltime_cadru))

    print(f"Procesare video... Salvare în '{cale_video_iesire}'")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Convertim cadrul la tonuri de gri
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Aplicăm un blur pentru a reduce zgomotul și a îmbunătăți detecția
        gray_blurred = cv2.medianBlur(gray, 5)

        # 3. Detectăm cercurile folosind Transformata Hough
        # Parametrii (dp, minDist, param1, param2) pot necesita ajustare în funcție de video
        detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                            param1=100, param2=30, minRadius=20, maxRadius=100)

        # 4. Desenăm cercurile detectate pe cadrul original
        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))
            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
                # Desenăm circumferința cercului
                cv2.circle(frame, (a, b), r, (0, 255, 0), 2)
                # Desenăm centrul cercului
                cv2.circle(frame, (a, b), 1, (0, 0, 255), 3)

        # Scriem cadrul procesat în fișierul de ieșire
        out.write(frame)

    # Eliberăm resursele
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Procesare video finalizată.")
    print("\n")


# --- Sarcina 2: Calculează și afișează histograma unei imagini ---
def sarcina_2(img_rgb):
    """Calculează și plotează histogramele pentru canalele R, G, B ale unei imagini."""
    print("--- Sarcina 2: Histograma Imaginii ---")

    culori = ('r', 'g', 'b')
    canale = cv2.split(img_rgb)

    plt.figure(figsize=(10, 5))
    plt.title("Histograma pe Canale de Culoare (RGB)")
    plt.xlabel("Intensitate Bins")
    plt.ylabel("Număr de Pixeli")

    for canal, culoare in zip(canale, culori):
        # cv2.calcHist([imagine], [canal], mask, [dimensiune_hist], [range])
        hist = cv2.calcHist([canal], [0], None, [256], [0, 256])
        plt.plot(hist, color=culoare)
        plt.xlim([0, 256])

    plt.legend(['Canal Roșu', 'Canal Verde', 'Canal Albastru'])
    plt.show()
    print("Histograma a fost calculată și afișată.")
    print("\n")


# --- Sarcina 3: Egalizarea histogramei ---
def sarcina_3(img_rgb):
    """Aplică egalizarea histogramei pe o imagine și afișează rezultatele."""
    print("--- Sarcina 3: Egalizarea Histogramei ---")

    # 1. Convertim la tonuri de gri
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 2. Aplicăm egalizarea histogramei
    img_egalizata = cv2.equalizeHist(img_gray)

    # Afișăm imaginile
    afiseaza_imagini([img_gray, img_egalizata], ["Gri Original", "Gri Egalizat"],
                     "Comparare Imagine Originală vs. Egalizată")

    # 3. Calculăm histogramele pentru ambele imagini (gri și egalizată)
    hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist_egalizata = cv2.calcHist([img_egalizata], [0], None, [256], [0, 256])

    # 4. Afișăm histogramele
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Histograma Imaginii Gri Originale")
    plt.plot(hist_gray)
    plt.xlim([0, 256])

    plt.subplot(1, 2, 2)
    plt.title("Histograma Imaginii Egalizate")
    plt.plot(hist_egalizata)
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()
    print("Egalizarea histogramei a fost aplicată și rezultatele afișate.")
    print("\n")


# --- Execuția principală ---
if __name__ == '__main__':
    # Sarcina 1: Necesită un fișier video local numit 'v1.mp4'
    # Puteți descărca un exemplu de video și redenumiți-l corespunzător.
    sarcina_1('v1.mp4')

    # Sarcini 2 & 3: Folosim o imagine de test
    URL_IMAGINE = "https://i.imgur.com/3_VA.jpg"  # O imagine cu contrast redus, bună pentru egalizare
    imaginea_mea = incarcare_imagine_test(URL_IMAGINE, nume_fisier="imagine_test_lab2.jpg")

    if imaginea_mea.any():  # Verificăm dacă imaginea nu este goală
        sarcina_2(imaginea_mea)
        sarcina_3(imaginea_mea)
    else:
        print("Scriptul nu a putut rula sarcinile 2 și 3 deoarece imaginea de test nu a fost încărcată.")

